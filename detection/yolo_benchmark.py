"""
yolo_benchmark.py
==================
Compare 2 architectures/versions de YOLO (YOLOv8n vs YOLOv10n, configurable)
sur la tâche de DÉTECTION binaire (cible / pas de cible, `single_cls=True`)
du dataset RADDet40K.

--------------------------------------------------------------------------
CORRECTIF PAR RAPPORT À LA VERSION PRÉCÉDENTE
--------------------------------------------------------------------------
La matrice de confusion générée automatiquement par Ultralytics
(`model.val()` -> `confusion_matrix.png`) a 3 limites documentées
(cf. issues Ultralytics #5995, #13440) :
  1. Elle est calculée au niveau BOÎTE, pas au niveau IMAGE : une image
     peut générer plusieurs boîtes, ce qui gonfle artificiellement les
     compteurs FP/FN par rapport à une évaluation "1 image = 1 décision"
     (celle utilisée pour CA-CFAR et RadarCNN dans le reste du projet).
  2. Le seuil de confiance utilisé pour la construire est TOUJOURS 0.25,
     quel que soit le `conf` passé à `model.val()` — donc pas forcément le
     point de fonctionnement qu'on veut réellement comparer à CA-CFAR.
  3. Elle ne peut jamais remplir la case "vrai négatif" (background prédit
     / background réel) : en détection, il n'existe pas d'instance
     "background" à faire correspondre à une absence de prédiction.

Cependant, ici, on connaît une information supplémentaire que le
validateur Ultralytics n'utilise pas : **l'absence (ou le vide) du
fichier .txt de labels prouve, par construction du dataset, que l'image
est H0 (bruit pur, aucune cible)**. On peut donc reconstruire une vraie
matrice 2x2 IMAGE-LEVEL (avec TP, FP, FN et surtout TN) :

    Vérité terrain (image-level) :
        - fichier label absent OU vide  -> H0 (pas de cible)
        - fichier label présent et non vide -> H1 (cible présente)

    Prédiction (image-level), à un seuil de confiance choisi CONF_THRESHOLD :
        - au moins une boîte avec conf >= CONF_THRESHOLD -> "cible détectée"
        - aucune boîte au-dessus du seuil -> "pas de cible détectée"

Cette matrice est directement comparable, cellule à cellule, à celle déjà
produite pour CA-CFAR (TN, FP, FN, TP sur les mêmes 20 001 images de test),
et permet de calculer un vrai Pfa (= FP / (FP+TN)) au sens radar du terme,
au lieu de se contenter des métriques boîte d'Ultralytics.

La matrice Ultralytics par défaut est conservée en complément (utile pour
le diagnostic boîte-par-boîte), mais la matrice IMAGE-LEVEL est désormais
la référence pour comparer YOLO à CA-CFAR / RadarCNN.
--------------------------------------------------------------------------

Structure attendue du dataset (identique à yolo.py) :
    <DATASET_DIR>/images/{train,val,test}/*.png
    <DATASET_DIR>/labels/{train,val,test}/*.txt   (absent ou vide = H0)
"""

import os
import shutil
import time
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.environ.get("RADAR_BASE_DIR", ".")
DATASET_DIR = os.environ.get("RADDET_DIR", os.path.join(BASE_DIR, "RADDet"))
YAML_PATH = os.path.join(BASE_DIR, "radar_binary_auto.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "yolo_benchmark_results")

# Seuil de confiance utilisé pour la matrice de confusion IMAGE-LEVEL
# (à aligner sur le point de fonctionnement souhaité ; 0.25 = défaut
# Ultralytics, mais peut être resserré pour se rapprocher du Pfa=10^-2
# utilisé pour CA-CFAR).
IMAGE_LEVEL_CONF_THRESHOLD = 0.25

# Repères CA-CFAR (rapport Swaidan, Table 4.2) pour comparaison directe
# dans les impressions console.
CFAR_REFERENCE = {"Pd": 0.9088, "Pfa": 50 / (50 + 1919)}  # TN=1919, FP=50

# Les 2 architectures/versions comparées
MODELS_TO_COMPARE = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv10n": "yolov10n.pt",
}

TRAIN_PARAMS = dict(
    epochs=50,
    imgsz=128,
    batch=64,
    optimizer="Adam",
    lr0=3e-4,
    weight_decay=1e-4,
    patience=10,
    single_cls=True,     # fusionne les 11 classes en 1 seule : détection pure

    # --- Augmentations désactivées : invalides sur un spectrogramme ---
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    scale=0.0,
    translate=0.05,

    device="cpu",         # remplacer par 0 si GPU disponible
    workers=8,
    verbose=True,
)


def write_yaml():
    yaml_content = f"""
path: {DATASET_DIR.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test

names:
  0: target  # Toutes les classes (0 à 10) sont fusionnées sous cette étiquette unique
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())


def train_and_validate(model_name, weights_file):
    print("\n" + "=" * 75)
    print(f">>> {model_name} : entraînement + validation (détection binaire)")
    print("=" * 75)

    model = YOLO(weights_file)
    start = time.time()
    model.train(name=f"{model_name}_Radar_Binary", data=YAML_PATH, **TRAIN_PARAMS)
    train_minutes = (time.time() - start) / 60

    # Validation Ultralytics standard (métriques boîte + matrice par défaut)
    val_results = model.val(data=YAML_PATH, split="test", imgsz=128, batch=64,
                             name=f"{model_name}_Radar_Binary_test")

    metrics_dict = val_results.results_dict
    save_dir = str(val_results.save_dir)

    return model_name, model, metrics_dict, save_dir, train_minutes


def collect_confusion_matrix(save_dir, model_name, output_dir):
    """Copie la matrice de confusion Ultralytics (boîte-level, conf=0.25 fixe,
    sans TN) — conservée à titre de diagnostic complémentaire uniquement."""
    src_normalized = os.path.join(save_dir, "confusion_matrix_normalized.png")
    src_raw = os.path.join(save_dir, "confusion_matrix.png")
    os.makedirs(output_dir, exist_ok=True)
    copied = []
    for src, tag in [(src_raw, "raw"), (src_normalized, "normalized")]:
        if os.path.isfile(src):
            dst = os.path.join(output_dir, f"confusion_matrix_boxlevel_{model_name}_{tag}.png")
            shutil.copy(src, dst)
            copied.append(dst)
            print(f"[yolo_benchmark] (boîte-level, référence) copiée -> {dst}")
    if not copied:
        print(f"[yolo_benchmark] ATTENTION : aucune matrice trouvée dans {save_dir}")
    return copied


# ----------------------------------------------------------------------
# Matrice de confusion IMAGE-LEVEL (avec TN) — le vrai correctif
# ----------------------------------------------------------------------
def build_image_level_ground_truth(dataset_dir, split):
    """
    Construit la vérité terrain image-level en s'appuyant sur la règle :
        - fichier label absent OU vide -> H0 (pas de cible), label = 0
        - fichier label présent et non vide -> H1 (cible présente), label = 1
    """
    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)

    ground_truth = {}  # chemin image -> 0/1
    valid_ext = (".png", ".jpg", ".jpeg")
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(valid_ext):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")

        has_target = False
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            has_target = len(lines) > 0

        ground_truth[os.path.join(images_dir, fname)] = 1 if has_target else 0

    n_h1 = sum(ground_truth.values())
    n_h0 = len(ground_truth) - n_h1
    print(f"[yolo_benchmark] Vérité terrain '{split}' : {len(ground_truth)} images "
          f"({n_h1} H1 / {n_h0} H0, via présence/absence de fichier label)")
    return ground_truth


def predict_image_level(model, ground_truth, conf_threshold, imgsz=128, batch_size=64):
    """Exécute l'inférence et réduit chaque image à une décision binaire :
    'cible détectée' si au moins une boîte a une confiance >= conf_threshold."""
    image_paths = list(ground_truth.keys())
    y_true = np.array([ground_truth[p] for p in image_paths])
    y_pred = []

    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        results = model.predict(chunk, imgsz=imgsz, conf=conf_threshold,
                                 verbose=False)
        for r in results:
            detected = r.boxes is not None and len(r.boxes) > 0
            y_pred.append(1 if detected else 0)

    return y_true, np.array(y_pred)


def compute_image_level_confusion(y_true, y_pred):
    """Retourne (TP, FN, FP, TN) avec la convention 1=target, 0=background."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    return tp, fn, fp, tn, cm


def plot_image_level_confusion_matrix(cm, model_name, out_path):
    """Trace la matrice 2x2 (target/background) avec TN désormais renseigné,
    au même format visuel (brut + normalisé) que les autres scripts du projet."""
    class_names = ["target", "background"]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, matrix, fmt, sub_title in [
        (axes[0], cm, "d", f"{model_name} — Effectifs (image-level)"),
        (axes[1], cm_norm, ".2%", f"{model_name} — Normalisée"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(sub_title)
        ax.set_xticks(range(2)); ax.set_yticks(range(2))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        thresh = matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[yolo_benchmark] Matrice de confusion IMAGE-LEVEL sauvegardée -> {out_path}")


def run_image_level_evaluation(model, model_name, output_dir, conf_threshold=IMAGE_LEVEL_CONF_THRESHOLD):
    ground_truth = build_image_level_ground_truth(DATASET_DIR, "test")
    y_true, y_pred = predict_image_level(model, ground_truth, conf_threshold)
    tp, fn, fp, tn, cm = compute_image_level_confusion(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = Pd
    pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0       # = Pfa, comparable à CA-CFAR
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n=== {model_name} — Évaluation IMAGE-LEVEL (conf={conf_threshold}) ===")
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"Précision={precision:.4f}  Rappel(Pd)={recall:.4f}  Pfa={pfa:.4f}  "
          f"F1={f1:.4f}  Accuracy={accuracy:.4f}")
    print(f"[Référence CA-CFAR] Pd={CFAR_REFERENCE['Pd']:.4f}  Pfa={CFAR_REFERENCE['Pfa']:.4f}")

    cm_path = os.path.join(output_dir, f"confusion_matrix_imagelevel_{model_name}.png")
    plot_image_level_confusion_matrix(cm, model_name, cm_path)

    return {
        "model": model_name, "conf_threshold": conf_threshold,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "precision": precision, "recall_pd": recall, "pfa": pfa,
        "f1": f1, "accuracy": accuracy,
    }


def build_comparison_table(all_box_metrics, all_image_metrics, output_dir):
    rows = []
    for model_name, m in all_box_metrics.items():
        precision = m.get("metrics/precision(B)", float("nan"))
        recall = m.get("metrics/recall(B)", float("nan"))
        map50 = m.get("metrics/mAP50(B)", float("nan"))
        map5095 = m.get("metrics/mAP50-95(B)", float("nan"))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({
            "model": model_name, "precision": precision, "recall": recall,
            "f1": f1, "map50": map50, "map50_95": map5095,
        })

    print("\n" + "=" * 90)
    print("TABLEAU COMPARATIF — MÉTRIQUES BOÎTE ULTRALYTICS (référence Ultralytics)")
    print("=" * 90)
    print(f"{'Modèle':<12} | {'Précision':>10} | {'Rappel':>8} | {'F1':>7} | "
          f"{'mAP@50':>8} | {'mAP@50-95':>10}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:<12} | {r['precision']:10.4f} | {r['recall']:8.4f} | "
              f"{r['f1']:7.4f} | {r['map50']:8.4f} | {r['map50_95']:10.4f}")

    print("\n" + "=" * 90)
    print("TABLEAU COMPARATIF — ÉVALUATION IMAGE-LEVEL (comparable à CA-CFAR / RadarCNN)")
    print("=" * 90)
    print(f"{'Modèle':<12} | {'Précision':>10} | {'Pd (Rappel)':>11} | {'Pfa':>8} | {'F1':>7} | {'Accuracy':>9}")
    print("-" * 90)
    for m in all_image_metrics:
        print(f"{m['model']:<12} | {m['precision']:10.4f} | {m['recall_pd']:11.4f} | "
              f"{m['pfa']:8.4f} | {m['f1']:7.4f} | {m['accuracy']:9.4f}")
    print(f"{'CA-CFAR':<12} | {'0.9970':>10} | {CFAR_REFERENCE['Pd']:11.4f} | "
          f"{CFAR_REFERENCE['Pfa']:8.4f} | {'0.9508':>7} | {'-':>9}")
    print("=" * 90)

    with open(os.path.join(output_dir, "comparison_table_boxlevel.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(output_dir, "comparison_table_imagelevel.json"), "w") as f:
        json.dump(all_image_metrics, f, indent=2)

    # Graphique en barres comparatif (image-level, le plus pertinent pour la thèse)
    metrics_to_plot = ["precision", "recall_pd", "f1", "accuracy"]
    labels_fr = ["Précision", "Pd (Rappel)", "F1", "Accuracy"]
    x = range(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(all_image_metrics):
        offsets = [xi + (i - 0.5) * width for xi in x]
        values = [m[k] for k in metrics_to_plot]
        ax.bar(offsets, values, width=width, label=m["model"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_fr)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison YOLOv8 vs YOLOv10 — Évaluation image-level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "comparison_barplot_imagelevel.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[yolo_benchmark] Graphique comparatif sauvegardé -> {fig_path}")

    return rows


def main():
    write_yaml()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_box_metrics = {}
    all_image_metrics = []
    all_times = {}

    for model_name, weights_file in MODELS_TO_COMPARE.items():
        name, model, metrics_dict, save_dir, train_minutes = train_and_validate(model_name, weights_file)
        all_box_metrics[name] = metrics_dict
        all_times[name] = train_minutes

        collect_confusion_matrix(save_dir, name, OUTPUT_DIR)  # référence boîte-level
        image_metrics = run_image_level_evaluation(model, name, OUTPUT_DIR)  # correctif : TN inclus
        all_image_metrics.append(image_metrics)

    build_comparison_table(all_box_metrics, all_image_metrics, OUTPUT_DIR)

    print("\nTemps d'entraînement :")
    for name, t in all_times.items():
        print(f"  {name}: {t:.1f} min")


if __name__ == "__main__":
    main()
