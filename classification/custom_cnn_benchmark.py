"""
custom_cnn_benchmark.py
=======================
CNN dédié pour la DISCRIMINATION DE CLASSE (11 modulations : Barker, Costas,
Frank, P1-P4, Poly, T1-T3) sur les spectrogrammes RADDet40K contenant une
cible (H1).

Ce script suit la structure de resnet_benchmark.py : mêmes fonctions
utilitaires (radar_common), mêmes données, mêmes pertes pondérées, même
optimiseur (AdamW, weight decay 1e-4), même arrêt anticipé (patience 10) et
même budget total d'époques (5 + 40). Le CNN est entraîné depuis zéro, sans
poids ImageNet, donc sans phase d'échauffement de la tête.

Architecture : 4 blocs strictement séquentiels, canaux 32 -> 64 -> 128 -> 256,
chacun Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2, puis AdaptiveAvgPool2d
et la même tête que les ResNet : Dropout, Linear, ReLU, BatchNorm1d, Dropout,
Linear à 11 sorties. Entrée : spectrogramme RGB en fausses couleurs, 3 canaux, tel que fourni par
radar_common (IMG_CHANNELS).
"""

import os
import torch
import torch.nn as nn

from radar_common import (
    set_seed, get_dataloaders, train_model, evaluate, IMG_CHANNELS,
    save_confusion_matrix, compute_and_report_metrics,
    append_metrics_to_leaderboard, print_leaderboard, CLASS_NAMES, NUM_CLASSES,
)

# ----------------------------------------------------------------------
# Configuration (mêmes noms et valeurs que resnet_benchmark.py)
# ----------------------------------------------------------------------
DATASET_DIR = os.environ.get("RADDET_DIR", "./RADDet")
OUTPUT_DIR = "custom_cnn_results"
BATCH_SIZE = 64
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 40
MAX_EPOCHS = HEAD_EPOCHS + FINE_TUNE_EPOCHS   # même budget que les ResNet
PATIENCE = 10
LR = 1e-3                                     # = LR_HEAD des ResNet
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNELS = (32, 64, 128, 256)
HIDDEN = 512
DROPOUT_1 = 0.4
DROPOUT_2 = 0.3


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class RadarCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=IMG_CHANNELS):
        super().__init__()
        chans = (in_channels,) + CHANNELS
        self.features = nn.Sequential(
            *[conv_block(chans[i], chans[i + 1]) for i in range(len(CHANNELS))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT_1),
            nn.Linear(CHANNELS[-1], HIDDEN),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN),
            nn.Dropout(DROPOUT_2),
            nn.Linear(HIDDEN, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(x)


def run_cnn(train_loader, val_loader, test_loader, class_weights, out_dir):
    name = "CNN"
    print(f"\n{'='*70}\n>>> Training {name}\n{'='*70}")
    model = RadarCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, DEVICE,
        scheduler=scheduler, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        model_name=name,
    )

    # ---- Final Evaluation ----
    _, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[{name}] Test Accuracy = {test_acc*100:.2f}%")

    cm_path = os.path.join(out_dir, f"confusion_matrix_{name}.png")
    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path,
                          title=f"{name} — Class Discrimination")

    metrics = compute_and_report_metrics(y_true, y_pred, y_prob, CLASS_NAMES, name)
    append_metrics_to_leaderboard(metrics, os.path.join(out_dir, "..", "leaderboard.json"))

    torch.save(model.state_dict(), os.path.join(out_dir, f"{name}_best.pt"))
    return metrics, history


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        DATASET_DIR, batch_size=BATCH_SIZE
    )

    dist = train_loader.dataset.class_distribution()
    counts = list(dist.values())
    if counts:
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 5:
            print(f"\n*** WARNING *** Extreme class imbalance detected (ratio "
                  f"{ratio:.1f}x): {dict(sorted(dist.items()))}\n"
                  "Please double-check your dataset label parsing!\n")

    metrics, _ = run_cnn(train_loader, val_loader, test_loader,
                         class_weights, OUTPUT_DIR)

    print("\n" + "=" * 78)
    print("CNN — résultat final")
    print("=" * 78)
    print(f"{'Model':<12} | {'Accuracy':>9} | {'F1-macro':>9} | {'AUC':>7} | {'Spread(pp)':>10}")
    print("-" * 78)
    print(f"{metrics['model']:<12} | {metrics['accuracy']*100:8.2f}% | "
          f"{metrics['f1_macro']*100:8.2f}% | {metrics['auc_ovr_macro']:.4f} | "
          f"{metrics['recall_spread_pp']:9.2f}")
    print("=" * 78)

    print_leaderboard(os.path.join(OUTPUT_DIR, "..", "leaderboard.json"))


if __name__ == "__main__":
    main()
