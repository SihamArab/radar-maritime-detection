"""
resnet_benchmark.py
====================
Compare 3 variantes de ResNet (ResNet18, ResNet34, ResNet50) sur la tâche
de DISCRIMINATION DE CLASSE (11 modulations : Barker, Costas, Frank, P1-P4,
Poly, T1-T3) sur les spectrogrammes RADDet40K contenant une cible (H1).
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

from radar_common import (
    set_seed, get_dataloaders, train_model, evaluate,
    save_confusion_matrix, compute_and_report_metrics,
    append_metrics_to_leaderboard, print_leaderboard, CLASS_NAMES, NUM_CLASSES,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_DIR = os.environ.get("RADDET_DIR", "./RADDet")
OUTPUT_DIR = "resnet_results"
BATCH_SIZE = 64
HEAD_EPOCHS = 5          
FINE_TUNE_EPOCHS = 40    
PATIENCE = 10
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4      
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_VARIANTS = {
    "ResNet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "ResNet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def get_discriminative_param_groups(model, base_lr):
    """
    Dynamically scales learning rates across the network depth.
    Works flawlessly for ResNet18, 34, and 50 by inspecting named modules.
    Early feature layers get lower LR, deeper layers get higher LR.
    """
    layers = [module for module in model.children()]
    fc_layer = layers[-1]
    backbone_layers = layers[:-1]
    
    num_backbone = len(backbone_layers)
    groups = []
    
    for i, layer in enumerate(backbone_layers):
        scale = 0.05 + (0.45 * (i / max(1, num_backbone - 1)))
        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * scale})
            
    fc_params = [p for p in fc_layer.parameters() if p.requires_grad]
    if fc_params:
        groups.append({"params": fc_params, "lr": base_lr * 1.0})
        
    return groups


def adapt_resnet_for_radar(model):
    """
    Adapts the stem of ResNet to the RGB spectrograms of radar_common
    (3 channels, pretrained weights kept) and prevents aggressive downsampling to preserve fine frequency signatures.
    """
    conv1 = model.conv1
    new_conv = nn.Conv2d(
        in_channels=conv1.in_channels, out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size, stride=1, 
        padding=conv1.padding, bias=(conv1.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:] = conv1.weight
        if conv1.bias is not None:
            new_conv.bias[:] = conv1.bias
    model.conv1 = new_conv
    
    model.maxpool = nn.Identity()
    return model


def build_resnet(variant_name, num_classes=NUM_CLASSES):
    ctor, weights = RESNET_VARIANTS[variant_name]
    model = ctor(weights=weights)
    
    model = adapt_resnet_for_radar(model)
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), 
        nn.Linear(in_features, out_features=512),  # Fixed typo here
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def run_one_resnet(variant_name, train_loader, val_loader, test_loader,
                    class_weights, out_dir):
    print(f"\n{'='*70}\n>>> Training {variant_name}\n{'='*70}")
    model = build_resnet(variant_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ---- Phase 1: Head Warm-up ----
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    model, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, DEVICE,
        max_epochs=HEAD_EPOCHS, patience=HEAD_EPOCHS, model_name=f"{variant_name}-head",
    )

    # ---- Phase 2: Full Fine-Tuning ----
    set_backbone_trainable(model, trainable=True)
    param_groups = get_discriminative_param_groups(model, LR_FINETUNE)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINE_TUNE_EPOCHS
    )
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, DEVICE,
        scheduler=scheduler, max_epochs=FINE_TUNE_EPOCHS, patience=PATIENCE,
        model_name=variant_name,
    )

    # ---- Final Evaluation ----
    _, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[{variant_name}] Test Accuracy = {test_acc*100:.2f}%")

    cm_path = os.path.join(out_dir, f"confusion_matrix_{variant_name}.png")
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path,
                           title=f"{variant_name} — Class Discrimination")

    metrics = compute_and_report_metrics(y_true, y_pred, y_prob, CLASS_NAMES, variant_name)
    append_metrics_to_leaderboard(metrics, os.path.join(out_dir, "..", "leaderboard.json"))

    torch.save(model.state_dict(), os.path.join(out_dir, f"{variant_name}_best.pt"))
    return metrics, history


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        DATASET_DIR, batch_size=BATCH_SIZE
    )

    train_ds = train_loader.dataset
    dist = train_ds.class_distribution()
    counts = list(dist.values())
    if counts:
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 5:
            print(f"\n*** WARNING *** Extreme class imbalance detected (ratio "
                  f"{ratio:.1f}x): {dict(sorted(dist.items()))}\n"
                  "Please double-check your dataset label parsing!\n")

    all_metrics = []
    for variant_name in RESNET_VARIANTS:
        metrics, _ = run_one_resnet(
            variant_name, train_loader, val_loader, test_loader,
            class_weights, OUTPUT_DIR,
        )
        all_metrics.append(metrics)

    print("\n" + "=" * 78)
    print("FINAL COMPARISON TABLE — ResNet18 vs ResNet34 vs ResNet50")
    print("=" * 78)
    print(f"{'Model':<12} | {'Accuracy':>9} | {'F1-macro':>9} | {'AUC':>7} | {'Spread(pp)':>10}")
    print("-" * 78)
    for m in sorted(all_metrics, key=lambda x: x["accuracy"], reverse=True):
        print(f"{m['model']:<12} | {m['accuracy']*100:8.2f}% | {m['f1_macro']*100:8.2f}% "
              f"| {m['auc_ovr_macro']:.4f} | {m['recall_spread_pp']:9.2f}")
    print("=" * 78)

    print_leaderboard(os.path.join(OUTPUT_DIR, "..", "leaderboard.json"))


if __name__ == "__main__":
    main()