"""
radar_common.py
================
Module utilitaire partagé par les scripts de benchmark :
    - yolo_benchmark.py         (détection : YOLOv8 vs YOLOv10)
    - resnet_benchmark.py       (discrimination de classe : ResNet18/34/50)
    - custom_cnn_benchmark.py   (discrimination de classe : CNN maison)
    - transformer_benchmark.py (discrimination de classe : Hybrid-ViT)

Objectif : centraliser le chargement du dataset RADDet40K, les
transformations, les métriques et la génération des matrices de confusion,
afin que les architectures soient évaluées EXACTEMENT dans les mêmes
conditions (même split, même normalisation, mêmes métriques) -> comparaison
scientifiquement valide.

--------------------------------------------------------------------------
CORRECTIF (v3) — instabilité de normalisation
--------------------------------------------------------------------------
Diagnostic sur un run ResNet18 réel : les stats calculées donnaient
std=[0.4955, 0.3547, 0.0127] -> le canal Bleu a un écart-type quasi nul.
`Normalize` divise par cet écart-type : n'importe quel pixel bleu qui
dévie même légèrement de la moyenne se retrouve amplifié d'un facteur
~78x, ce qui produit des valeurs d'entrée extrêmes et déstabilise
l'entraînement (particulièrement grave lors du fine-tuning d'un ResNet
pré-entraîné, dont les statistiques de BatchNorm sont calibrées pour des
entrées à variance ~1). Résultat observé : la train accuracy elle-même
plafonne très bas (le réseau n'arrive même pas à sur-apprendre le train
set), symptôme classique d'une instabilité numérique plutôt que d'un
problème de difficulté de la tâche.

Deux changements apportés :
  1. `compute_or_load_norm_stats` plancher désormais l'écart-type par
     canal à `std_floor` (0.05 par défaut) au lieu de `1e-8` : un canal
     quasi-constant reste quasi-constant après normalisation (valeurs
     proches de 0), au lieu d'exploser.
  2. `suppress_background` (nouveau paramètre, propagé depuis
     `get_dataloaders`) permet de désactiver `BackgroundSuppression` pour
     tester si le seuillage Otsu détruit les structures fines qui
     distinguent les classes proches (Frank/P1-P4/Poly/T1). Le nom du
     fichier de cache change selon ce flag pour ne jamais mélanger des
     statistiques calculées avec/sans suppression.
--------------------------------------------------------------------------

Auteur : pipeline généré pour le PFE "Détection de Cibles Radar en Milieu
Maritime" (ENSTA Bretagne / UBO), en complément du pipeline CA-CFAR déjà
finalisé.
"""

import os
import json
import random
import time
import re
from collections import Counter

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Constantes globales (alignées sur le pipeline CA-CFAR / RadarCNN déjà
# implémenté : 128x128, 11 classes de modulation, split RADDet40K).
# ----------------------------------------------------------------------
IMG_SIZE = 128
IMG_CHANNELS = 3  # Les PNG RADDet40K sont enregistrés en fausses couleurs
                  # (colormap type jet/viridis) : on les charge en RGB,
                  # EXACTEMENT comme le fait Ultralytics/YOLO nativement.
CLASS_NAMES = ["Barker", "Costas", "Frank", "P1", "P2", "P3", "P4",
               "Poly", "T1", "T2", "T3"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42


def set_seed(seed=SEED):
    """Fixe toutes les graines aléatoires pour la reproductibilité des runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RadDetClassificationDataset(Dataset):
    """
    Dataset de classification pour la discrimination des 11 classes de
    modulation de RADDet40K.

    Deux structures de dossiers sont supportées (auto-détection) :

    1) Format YOLO :
           <root>/images/<split>/*.png
           <root>/labels/<split>/*.txt   (une ligne "class cx cy w h" par cible)
       -> Seules les images H1 (label non vide) sont conservées. Si
       plusieurs objets sont présents, la classe majoritaire (mode) sert
       d'étiquette image-niveau.

    2) Format "ImageFolder" classique :
           <root>/<split>/<nom_de_classe>/*.png

    Le format est déterminé automatiquement en cherchant un dossier
    `labels/<split>` ; à défaut, on suppose le format ImageFolder.
    """

    def __init__(self, root, split, transform=None, class_names=CLASS_NAMES):
        self.root = root
        self.split = split
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        images_dir = os.path.join(root, "images", split)
        labels_dir = os.path.join(root, "labels", split)

        self.samples = []  # liste de (chemin_image, label_int)

        if os.path.isdir(labels_dir):
            self._build_from_yolo_format(images_dir, labels_dir)
        else:
            self._build_from_imagefolder_format(os.path.join(root, split))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Aucun échantillon H1 trouvé pour le split '{split}' dans {root}. "
                "Vérifiez l'arborescence du dataset (voir docstring de la classe)."
            )

    def _build_from_yolo_format(self, images_dir, labels_dir):
        valid_ext = (".png", ".jpg", ".jpeg")
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(valid_ext):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(label_path):
                continue  # pas de label -> considéré comme H0 (bruit pur), exclu
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue  # fichier vide -> H0, exclu de la classification
            class_ids = [int(float(l.split()[0])) for l in lines]
            # Classe majoritaire si plusieurs cibles dans la même image
            majority_class = Counter(class_ids).most_common(1)[0][0]
            if 0 <= majority_class < NUM_CLASSES:
                self.samples.append(
                    (os.path.join(images_dir, fname), majority_class)
                )

    def _build_from_imagefolder_format(self, split_dir):
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir) or class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_distribution(self):
        labels = [l for _, l in self.samples]
        return Counter(labels)


# ----------------------------------------------------------------------
# Augmentation : SpecAugment (Park et al., 2019) adapté aux spectrogrammes
# radar. On évite volontairement tout flip horizontal/vertical ou rotation
# géométrique (non pertinent pour un plan temps-fréquence). Seuls le
# masquage temps/fréquence et un bruit gaussien léger sont utilisés.
# ----------------------------------------------------------------------
class SpecAugment(object):
    """Masquage aléatoire de bandes temporelles et fréquentielles + bruit gaussien."""

    def __init__(self, freq_mask_param=12, time_mask_param=12,
                 n_freq_masks=1, n_time_masks=1, noise_std=0.02, p=0.5):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.noise_std = noise_std
        self.p = p

    def __call__(self, tensor):
        # tensor: [C, H, W] normalisé (C=3 pour les spectrogrammes RGB)
        if random.random() > self.p:
            return tensor
        c, h, w = tensor.shape
        out = tensor.clone()
        mean_val = out.mean(dim=(1, 2), keepdim=True)  # moyenne par canal
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, h - f))
            out[:, f0:f0 + f, :] = mean_val
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, w - t))
            out[:, :, t0:t0 + t] = mean_val
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        return out


# ----------------------------------------------------------------------
# Normalisation : calculée une seule fois sur le train set puis réutilisée
# à l'identique pour toutes les architectures (comparaison équitable).
# ----------------------------------------------------------------------
def otsu_threshold(gray_array):
    """
    Seuil d'Otsu (1979) : trouve automatiquement le seuil qui sépare le mieux
    2 populations de pixels (ici : fond/bruit vs. signal). Implémentation
    numpy pure, sans dépendance à OpenCV/scikit-image.
    """
    hist, _ = np.histogram(gray_array.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, best_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            threshold = t
    return float(threshold)


class BackgroundSuppression(object):
    """
    Suppression du fond par seuillage adaptatif (Otsu), transform déterministe
    appliquée systématiquement (train ET eval) QUAND `suppress_background=True`
    est passé plus haut dans la pipeline (cf. build_transforms /
    get_dataloaders). Conçue à l'origine pour la détection binaire (isoler le
    signal du bruit de fond avant classification cible/pas-cible) ; peut être
    désactivée pour la classification fine 11-classes si elle s'avère
    détruire les structures fines qui distinguent les codes proches
    (Frank/P1-P4/Poly/T1) — cf. note en tête de fichier.
    """

    def __call__(self, pil_img):
        rgb = np.array(pil_img).astype(np.uint8)          # [H, W, 3]
        gray = np.array(pil_img.convert("L")).astype(np.float64)
        t = otsu_threshold(gray)
        mask = gray > t
        out = rgb.copy()
        out[~mask] = 0
        return Image.fromarray(out)


def compute_or_load_norm_stats(root, cache_path=None, max_samples=2000,
                                suppress_background=True, std_floor=0.05):
    """Calcule (mean, std) PAR CANAL (RGB) sur un sous-échantillon du train
    set, ou charge le cache. Le nom du cache dépend de `suppress_background`
    pour ne jamais réutiliser des stats calculées avec un pré-traitement
    différent.

    CORRECTIF v3 : `std_floor` plafonne un écart-type pathologiquement
    faible (canal quasi-constant) à une valeur plancher raisonnable, pour
    éviter qu'une division par un écart-type quasi nul n'amplifie le bruit
    numérique d'un facteur x50-x100 lors de la Normalize -> instabilité
    d'entraînement, surtout dommageable en fine-tuning d'un backbone
    pré-entraîné.
    """
    if cache_path is None:
        cache_path = "norm_stats_v3_bg.json" if suppress_background else "norm_stats_v3_nobg.json"

    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    import torchvision.transforms as T
    ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if suppress_background:
        ops.append(BackgroundSuppression())
    ops.append(T.ToTensor())
    pre_tf = T.Compose(ops)

    ds = RadDetClassificationDataset(root, "train", transform=pre_tf)
    n = min(max_samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    sum_c = torch.zeros(IMG_CHANNELS)
    sumsq_c = torch.zeros(IMG_CHANNELS)
    n_pixels = 0
    for i in idxs:
        img, _ = ds[i]  # [C, H, W]
        sum_c += img.sum(dim=(1, 2))
        sumsq_c += (img ** 2).sum(dim=(1, 2))
        n_pixels += img.shape[1] * img.shape[2]
    mean = (sum_c / n_pixels)
    var = (sumsq_c / n_pixels) - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-8))

    floored = std < std_floor
    if floored.any():
        print(f"[radar_common] ATTENTION : écart-type quasi nul détecté sur le(s) "
              f"canal(aux) {floored.nonzero().flatten().tolist()} (std brut="
              f"{std.tolist()}) -> plancher appliqué à {std_floor} pour éviter "
              f"une amplification numérique lors de la Normalize.")
    std = torch.clamp(std, min=std_floor)

    mean, std = mean.tolist(), std.tolist()
    with open(cache_path, "w") as f:
        json.dump({"mean": mean, "std": std, "suppress_background": suppress_background}, f)
    return mean, std


def build_transforms(mean, std, train=True, augment=True, suppress_background=True):
    import torchvision.transforms as T
    ops = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if suppress_background:
        ops.append(BackgroundSuppression())
    ops += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    if train and augment:
        ops.append(SpecAugment())
    return T.Compose(ops)


def get_dataloaders(root, batch_size=64, num_workers=4, augment=True,
                     suppress_background=True):
    """Construit les 3 DataLoaders (train/val/test) + poids de classes."""
    mean, std = compute_or_load_norm_stats(root, suppress_background=suppress_background)
    mean_str = ", ".join(f"{m:.4f}" for m in mean)
    std_str = ", ".join(f"{s:.4f}" for s in std)
    print(f"[radar_common] Normalisation (par canal RGB, suppress_background="
          f"{suppress_background}) -> mean=[{mean_str}]  std=[{std_str}]")

    train_tf = build_transforms(mean, std, train=True, augment=augment,
                                 suppress_background=suppress_background)
    eval_tf = build_transforms(mean, std, train=False, augment=False,
                                suppress_background=suppress_background)

    train_ds = RadDetClassificationDataset(root, "train", transform=train_tf)
    val_ds = RadDetClassificationDataset(root, "val", transform=eval_tf)
    test_ds = RadDetClassificationDataset(root, "test", transform=eval_tf)

    print(f"[radar_common] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print(f"[radar_common] Distribution train: {train_ds.class_distribution()}")
    if len(val_ds) > len(train_ds):
        print(f"[radar_common] ATTENTION : le split 'val' ({len(val_ds)}) est plus "
              f"grand que 'train' ({len(train_ds)}). Vérifie l'arborescence "
              f"images/labels train/val -- ce n'est pas forcément une erreur, "
              f"mais c'est inhabituel.")

    labels = [l for _, l in train_ds.samples]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(NUM_CLASSES), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights


# ----------------------------------------------------------------------
# Boucle d'entraînement générique avec early stopping.
# ----------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                 scheduler=None, max_epochs=50, patience=10, model_name="model",
                 checkpoint_path=None, log_every=20):
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_epoch = 1

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[{model_name}] Reprise depuis le checkpoint à l'epoch {start_epoch}.")

    n_batches = len(train_loader)
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()
        print(f"[{model_name}] epoch {epoch:03d}/{max_epochs} — début "
              f"({n_batches} batches, {len(train_loader.dataset)} images)", flush=True)

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

            if batch_idx % log_every == 0 or batch_idx == n_batches:
                elapsed = time.time() - epoch_start
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                print(f"[{model_name}]   batch {batch_idx:04d}/{n_batches} "
                      f"loss={running_loss/total:.4f} acc={correct/total:.4f} "
                      f"({elapsed:.0f}s écoulées, ETA epoch ~{eta:.0f}s)", flush=True)

        train_loss = running_loss / total
        train_acc = correct / total

        print(f"[{model_name}] epoch {epoch:03d} — validation en cours "
              f"({len(val_loader.dataset)} images)...", flush=True)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] epoch {epoch:03d}/{max_epochs} TERMINÉ — "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-epoch_start:.0f}s)", flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if checkpoint_path:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            }, checkpoint_path)

        if epochs_no_improve >= patience:
            print(f"[{model_name}] Early stopping à l'epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return running_loss / total, correct / total, y_true, y_pred, y_prob


# ----------------------------------------------------------------------
# Matrice de confusion + métriques détaillées.
# ----------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, matrix, fmt, sub_title in [
        (axes[0], cm, "d", f"{title} — Effectifs"),
        (axes[1], cm_norm, ".2f", f"{title} — Normalisée (rappel par classe)"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(sub_title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[radar_common] Matrice de confusion sauvegardée -> {out_path}")
    return cm


# Regroupement littérature : Barker/Costas/T2/T3 ont des signatures
# temps-fréquence nettement distinctes (compactes ou à forte pente) ; les
# codes polyphase (Frank/P1-P4/Poly) et le polytime T1 sont mathématiquement
# proches (rampes de phase multi-segments) et documentés comme confusables
# même à haut SNR (Kong et al., 2018, IEEE Access).
EASY_CLASSES = ["Barker", "Costas", "T2", "T3"]
HARD_CLASSES = ["Frank", "P1", "P2", "P3", "P4", "Poly", "T1"]

SNR_FILENAME_PATTERNS = [
    re.compile(r"snr[_-]?(-?\d+)", re.IGNORECASE),
    re.compile(r"[_-](-?\d+)db", re.IGNORECASE),
]


def try_parse_snr_from_filename(path):
    """Essaie d'extraire la valeur de SNR (en dB) depuis le nom de fichier."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for pattern in SNR_FILENAME_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def get_snr_values(dataset):
    """Retourne la liste des SNR (ou None) pour chaque échantillon du dataset."""
    return [try_parse_snr_from_filename(p) for p, _ in dataset.samples]


# ----------------------------------------------------------------------
# SNR exact via les fichiers JSON de métadonnées.
# ----------------------------------------------------------------------
METADATA_DIR_CANDIDATES = ["metadata", "annotations", "meta", "json", "labels_json"]

_metadata_dir_cache = {}


def find_metadata_dir(root, split):
    """Auto-détecte le dossier contenant les .json pour un split donné."""
    key = (root, split)
    if key in _metadata_dir_cache:
        return _metadata_dir_cache[key]
    found = None
    for cand in METADATA_DIR_CANDIDATES:
        d = os.path.join(root, cand, split)
        if os.path.isdir(d):
            found = d
            break
    _metadata_dir_cache[key] = found
    return found


def load_json_metadata(root, split, stem):
    """Charge le .json de métadonnées correspondant à un échantillon."""
    d = find_metadata_dir(root, split)
    if d is None:
        return None
    path = os.path.join(d, stem + ".json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0] if data else None
        return data
    except (json.JSONDecodeError, IndexError, OSError):
        return None


def get_snr_values_from_json(dataset, root, split):
    """Equivalent de get_snr_values() mais lit le SNR EXACT depuis les
    fichiers .json de métadonnées."""
    values = []
    n_found = 0
    for path, _ in dataset.samples:
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = load_json_metadata(root, split, stem)
        if meta is not None and "SNR" in meta:
            values.append(float(meta["SNR"]))
            n_found += 1
        else:
            values.append(None)
    d = find_metadata_dir(root, split)
    if d is None:
        print(f"[radar_common] Aucun dossier de métadonnées trouvé parmi "
              f"{METADATA_DIR_CANDIDATES} pour le split '{split}'.")
    else:
        print(f"[radar_common] Métadonnées JSON trouvées dans {d} "
              f"({n_found}/{len(values)} SNR exacts récupérés).")
    return values


def report_easy_vs_hard_groups(y_true, y_pred, class_names, model_name):
    """Accuracy séparée pour le groupe de classes 'faciles' vs. 'difficiles'."""
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    easy_idx = {name_to_idx[c] for c in EASY_CLASSES if c in name_to_idx}
    hard_idx = {name_to_idx[c] for c in HARD_CLASSES if c in name_to_idx}

    def group_accuracy(idx_set):
        mask = np.isin(y_true, list(idx_set))
        if mask.sum() == 0:
            return float("nan"), 0
        return float((y_pred[mask] == y_true[mask]).mean()), int(mask.sum())

    easy_acc, n_easy = group_accuracy(easy_idx)
    hard_acc, n_hard = group_accuracy(hard_idx)
    print(f"\n[{model_name}] Accuracy par groupe (cf. littérature LPI radar) :")
    print(f"  Classes séparables (Barker/Costas/T2/T3)         : {easy_acc*100:.2f}% (n={n_easy})")
    print(f"  Classes polyphase/polytime proches (Frank/P1-P4/Poly/T1) : {hard_acc*100:.2f}% (n={n_hard})")
    return {"easy_group_acc": easy_acc, "hard_group_acc": hard_acc,
            "n_easy": n_easy, "n_hard": n_hard}


def report_snr_stratified(y_true, y_pred, snr_values, model_name,
                           bins=(-20, -15, -10, -5, 0, 5, 10, 15, 20)):
    """Recall stratifié par tranche de SNR."""
    snr_arr = np.array([s if s is not None else np.nan for s in snr_values])
    valid = ~np.isnan(snr_arr)
    if valid.sum() == 0:
        print(f"\n[{model_name}] Pas de SNR détecté -> évaluation stratifiée non disponible.")
        return None

    print(f"\n[{model_name}] Rappel (accuracy) par tranche de SNR "
          f"({valid.sum()}/{len(snr_values)} échantillons avec SNR détecté) :")
    print(f"{'Tranche SNR (dB)':<18} | {'Accuracy':>9} | {'n':>6}")
    print("-" * 40)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = valid & (snr_arr >= lo) & (snr_arr < hi)
        n = int(mask.sum())
        acc = float((y_pred[mask] == y_true[mask]).mean()) if n > 0 else float("nan")
        rows.append({"snr_bin": f"[{lo},{hi})", "accuracy": acc, "n": n})
        acc_str = f"{acc*100:.2f}%" if n > 0 else "n/a"
        print(f"[{lo:>4},{hi:>4})       | {acc_str:>9} | {n:>6}")
    return rows


def compute_and_report_metrics(y_true, y_pred, y_prob, class_names, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    try:
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                 labels=list(range(len(class_names))))
    except ValueError:
        auc_ovr = float("nan")

    all_labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=all_labels,
                                    target_names=class_names,
                                    output_dict=True, zero_division=0)
    per_class_recall = {c: report[c]["recall"] for c in class_names}
    spread_pp = (max(per_class_recall.values()) - min(per_class_recall.values())) * 100

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_ovr_macro": auc_ovr,
        "per_class_recall": per_class_recall,
        "recall_spread_pp": spread_pp,
    }
    print(f"\n=== Résultats {model_name} ===")
    print(f"Accuracy         : {acc*100:.2f}%")
    print(f"F1-macro         : {f1_macro*100:.2f}%")
    print(f"AUC (OvR macro)  : {auc_ovr:.4f}")
    print(f"Spread rappel    : {spread_pp:.2f} pp (max-min sur les 11 classes)")
    print(classification_report(y_true, y_pred, labels=all_labels,
          target_names=class_names, zero_division=0))
    return metrics


def append_metrics_to_leaderboard(metrics, leaderboard_path="leaderboard.json"):
    """Empile les métriques dans un fichier JSON partagé pour la comparaison
    finale inter-architectures."""
    board = []
    if os.path.isfile(leaderboard_path):
        with open(leaderboard_path) as f:
            board = json.load(f)
    board = [m for m in board if m["model"] != metrics["model"]]
    board.append(metrics)
    with open(leaderboard_path, "w") as f:
        json.dump(board, f, indent=2)
    print(f"[radar_common] Leaderboard mis à jour -> {leaderboard_path}")


def print_leaderboard(leaderboard_path="leaderboard.json"):
    if not os.path.isfile(leaderboard_path):
        print("Aucun leaderboard trouvé pour le moment.")
        return
    with open(leaderboard_path) as f:
        board = json.load(f)
    board = sorted(board, key=lambda m: m["accuracy"], reverse=True)
    print("\n" + "=" * 78)
    print(f"{'Modèle':<22} | {'Accuracy':>9} | {'F1-macro':>9} | {'AUC':>7} | {'Spread(pp)':>10}")
    print("-" * 78)
    for m in board:
        print(f"{m['model']:<22} | {m['accuracy']*100:8.2f}% | {m['f1_macro']*100:8.2f}% "
              f"| {m['auc_ovr_macro']:.4f} | {m['recall_spread_pp']:9.2f}")
    print("=" * 78)
    if board:
        best = board[0]
        print(f"\n-> Meilleure architecture pour la discrimination de classe : {best['model']}")
