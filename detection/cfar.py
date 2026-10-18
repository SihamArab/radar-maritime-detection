import os                      # Module pour manipuler les chemins de fichiers et le système de fichiers
import glob                    # Module pour lister les fichiers correspondant à un motif (ex: *.png)
import numpy as np             # Bibliothèque de calcul numérique (tableaux, opérations matricielles)
import cv2                     # OpenCV : lecture d'images et filtrage 2D (convolution)
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Métriques d'évaluation (ROC, AUC, accuracy, matrice de confusion)
import matplotlib.pyplot as plt  # Bibliothèque de tracé pour générer les figures (ROC, matrice de confusion)

# ==========================================
# 1. PARAMETERS (EXACTLY FROM THE PAPER)
# ==========================================
DATASET_PATH = "./RADDet"       # Chemin racine du dataset RADDet (contient images/ et labels/)
SPLITS = ["train", "val", "test"]  # Process all three splits automatically  # Liste des sous-ensembles à traiter automatiquement

EPSILON = 1e-6   # Petite constante pour éviter log(0) lors de la compression logarithmique
N_G = 2      # Guard half-width  # Demi-largeur de la fenêtre de garde (cellules exclues autour de la cellule sous test)
N_T = 8      # Reference half-width  # Demi-largeur de la fenêtre totale (référence + garde)
P_FA = 1e-2  # Target False Alarm Rate (0.01)  # Probabilité de fausse alarme visée (Pfa), utilisée pour calculer le seuil

def calculate_alpha(n_g, n_t, p_fa):
    # Calcule le facteur d'échelle alpha (seuil multiplicatif) et le nombre de cellules de référence
    total_window_side = 2 * n_t + 1          # Taille totale du côté de la fenêtre (garde + référence + cellule testée)
    guard_window_side = 2 * n_g + 1          # Taille du côté de la fenêtre de garde (zone exclue au centre)
    n_ref = (total_window_side ** 2) - (guard_window_side ** 2)  # Nombre de cellules de référence = aire totale - aire de garde
    alpha = n_ref * (p_fa ** (-1.0 / n_ref) - 1)  # Formule CA-CFAR classique pour obtenir alpha à partir de Pfa et n_ref
    return alpha, n_ref              # Retourne le couple (alpha, nombre de cellules de référence)

ALPHA, N_REF = calculate_alpha(N_G, N_T, P_FA)   # Calcule effectivement alpha et n_ref avec les paramètres définis plus haut (Pfa nominal = 1e-2)
print(f"Configured: {N_REF} Reference Cells | Scaling factor (Alpha): {ALPHA:.4f}")  # Affiche la configuration CFAR utilisée

# ==========================================
# 2. RADBOX COORDINATE PARSER
# ==========================================
def parse_raddet_labels(label_path, shape=(128, 128)):
    """
    Parses normalized YOLO/RADDet bounding boxes from text files 
    and constructs an exact 2D pixel truth mask.
    """
    mask = np.zeros(shape, dtype=np.int32)   # Initialise le masque de vérité terrain à zéro (aucune cible par défaut)
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:  # Si le fichier de labels n'existe pas ou est vide
        return mask, 0 # Empty file = Noise (H0)   # On retourne un masque vide et l'hypothèse H0 (bruit seul, pas de cible)
        
    with open(label_path, 'r') as f:         # Ouvre le fichier de labels en lecture
        for line in f:                       # Parcourt chaque ligne (chaque ligne = une boîte englobante)
            parts = line.strip().split()     # Supprime les espaces superflus et découpe la ligne en champs
            if len(parts) < 5:               # Vérifie qu'il y a bien au moins 5 champs (classe + 4 coordonnées)
                continue                     # Sinon on ignore cette ligne mal formée
            # YOLO Format: class, x_center, y_center, width, height (normalized)
            _, x_c, y_c, w, h = map(float, parts)  # Extrait les coordonnées normalisées (on ignore la classe avec _)
            
            # Map normalized scale back to 128x128 pixel indices
            x1 = int((x_c - w / 2) * shape[1])   # Coordonnée x du coin gauche de la boîte, convertie en pixels
            x2 = int((x_c + w / 2) * shape[1])   # Coordonnée x du coin droit de la boîte, convertie en pixels
            y1 = int((y_c - h / 2) * shape[0])   # Coordonnée y du coin haut de la boîte, convertie en pixels
            y2 = int((y_c + h / 2) * shape[0])   # Coordonnée y du coin bas de la boîte, convertie en pixels
            
            # Clip limits to safely stay inside image bounds
            x1, x2 = max(0, x1), min(shape[1], x2)  # Borne x1/x2 pour rester dans les limites de l'image
            y1, y2 = max(0, y1), min(shape[0], y2)  # Borne y1/y2 pour rester dans les limites de l'image
            
            mask[y1:y2, x1:x2] = 1           # Marque la zone de la boîte englobante comme cible (1) dans le masque
            
    return mask, 1 if np.any(mask) else 0    # Retourne le masque complet et l'étiquette frame (1 si au moins une cible présente)

# ==========================================
# 3. ADVANCED PROCESSING ENGINE
# ==========================================
def process_frame_cells(energy_map, label_path, n_g, n_t, n_ref):
    """
    Evaluates matrix profiles using real ground truth coordinate masks.
    """
    kernel_side = 2 * n_t + 1                # Taille du côté du noyau de convolution (fenêtre totale CFAR)
    kernel = np.ones((kernel_side, kernel_side), dtype=np.float32)  # Crée un noyau rempli de 1 (toutes les cellules incluses au départ)
    
    start_g = n_t - n_g                      # Indice de début de la zone de garde à l'intérieur du noyau
    end_g = n_t + n_g + 1                    # Indice de fin de la zone de garde à l'intérieur du noyau
    kernel[start_g:end_g, start_g:end_g] = 0 # Met à zéro la zone de garde (cellule testée + voisinage immédiat exclus du bruit)
    kernel /= n_ref                          # Normalise le noyau par le nombre de cellules de référence (moyenne du bruit)
    
    # Estimate noise floor and pixel-wise SNR
    local_noise_mean = cv2.filter2D(energy_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)  # Convolue la carte d'énergie avec le noyau pour estimer le niveau de bruit local
    local_noise_mean = np.where(local_noise_mean == 0, 1e-6, local_noise_mean)  # Évite les divisions par zéro en remplaçant les zéros par une valeur infime
    snr_matrix = energy_map / local_noise_mean  # Calcule le rapport signal/bruit (SNR) pixel par pixel
    
    # Restrict to valid interior region to avoid edge artifacts
    valid_interior = np.zeros_like(snr_matrix, dtype=bool)  # Crée un masque booléen initialisé à False partout
    valid_interior[n_t:-n_t, n_t:-n_t] = True  # Active uniquement la zone intérieure valide (évite les artefacts de bord dus à la convolution)
    
    # Parse coordinates from labels
    cell_true_matrix, frame_true = parse_raddet_labels(label_path, shape=energy_map.shape)  # Récupère le masque de vérité terrain et l'étiquette frame associée

    # Frame level predictions
    peak_snr = np.max(snr_matrix[valid_interior])  # Cherche le pic maximal de SNR dans la zone valide (score au niveau frame)
    frame_pred = 1 if peak_snr > ALPHA else 0       # Décision binaire frame : cible détectée si le pic dépasse le seuil alpha (Pfa nominal)
    
    # Flatten arrays extracted strictly from the valid calculation bounds
    flat_scores = snr_matrix[valid_interior].flatten()      # Aplati les scores SNR de la zone valide en un vecteur 1D
    flat_truths = cell_true_matrix[valid_interior].flatten()  # Aplati les vérités terrain correspondantes en un vecteur 1D
    
    return frame_true, frame_pred, flat_scores, flat_truths  # Retourne vérité/prédiction au niveau frame + scores/vérités au niveau cellule

# ==========================================
# 4. MAIN RUNNER
# ==========================================
def main():
    # Loop over train, val, and test datasets automatically
    for split in SPLITS:                     # Boucle sur chaque sous-ensemble (train, val, test)
        image_dir = os.path.join(DATASET_PATH, "images", split)  # Construit le chemin du dossier d'images pour ce split
        label_dir = os.path.join(DATASET_PATH, "labels", split)  # Construit le chemin du dossier de labels pour ce split
        
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # Liste triée de tous les fichiers PNG du split
        
        if not image_files:                  # Si aucun fichier image trouvé dans ce dossier
            print(f"\nNo images found in {image_dir}. Skipping this split.")  # Prévient l'utilisateur
            continue                         # Passe au split suivant sans traitement

        y_frame_true = []                    # Liste des vérités terrain au niveau frame (pour ce split)
        y_frame_pred = []                    # Liste des prédictions au niveau frame (pour ce split, calculées avec ALPHA nominal)
        
        all_cell_scores = []                 # Liste des scores SNR au niveau cellule (non utilisée pour le ROC dans cette version, gardée pour la matrice de confusion)
        all_cell_truths = []                 # Liste des vérités terrain au niveau cellule (idem, gardée pour compatibilité/statistiques)
        
        print(f"\nProcessing {len(image_files)} frames for {split.upper()} split...")  # Indique le nombre de frames à traiter
        
        for img_path in image_files:         # Parcourt chaque image du split
            base_name = os.path.basename(img_path)         # Récupère le nom de fichier sans le chemin
            file_id = os.path.splitext(base_name)[0]        # Récupère l'identifiant du fichier sans extension
            lbl_path = os.path.join(label_dir, f"{file_id}.txt")  # Construit le chemin du fichier de labels correspondant
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Charge l'image en niveaux de gris
            if img is None:                  # Si la lecture de l'image a échoué
                continue                     # Ignore ce fichier et passe au suivant
                
            img_float = img.astype(np.float32) / 255.0       # Convertit l'image en float et normalise entre 0 et 1
            log_compressed = np.log10(img_float + EPSILON)   # Applique une compression logarithmique (avec epsilon pour éviter log(0))
            energy_map = np.square(log_compressed)            # Calcule la carte d'énergie en élevant au carré la valeur log-compressée
            
            # Process arrays using genuine labels
            f_true, f_pred, c_scores, c_truths = process_frame_cells(energy_map, lbl_path, N_G, N_T, N_REF)  # Applique le pipeline CFAR sur cette frame (au seuil ALPHA nominal)
            
            y_frame_true.append(f_true)      # Ajoute la vérité terrain frame à la liste globale
            y_frame_pred.append(f_pred)      # Ajoute la prédiction frame (seuil nominal) à la liste globale
            
            all_cell_scores.append(c_scores)  # Ajoute les scores cellule de cette frame à la liste globale
            all_cell_truths.append(c_truths)  # Ajoute les vérités cellule de cette frame à la liste globale

        print("\n" + "="*50)                 # Ligne de séparation visuelle dans la console
        print(f" CA-CFAR FRAME-LEVEL EVALUATION ({split.upper()} SET)")  # Titre de la section d'évaluation frame-level
        print("="*50)                        # Ligne de séparation visuelle
        print(f"Accuracy Score: {accuracy_score(y_frame_true, y_frame_pred):.4f}")  # Affiche l'exactitude (accuracy) au niveau frame, au seuil nominal
        print("\nDetailed Performance Matrix:")  # Titre du rapport détaillé
        print(classification_report(y_frame_true, y_frame_pred, target_names=["Noise (H0)", "Target (H1)"], zero_division=0))  # Affiche précision/rappel/F1 par classe (H0/H1)

        # --- GENERATING THE CONFUSION MATRIX ---
        cm = confusion_matrix(y_frame_true, y_frame_pred)  # Calcule la matrice de confusion frame-level (au seuil nominal)
        print("Raw Confusion Matrix:")        # Titre avant l'affichage brut
        print(cm)                             # Affiche la matrice de confusion sous forme de tableau numérique
        
        # Save a clean, visual confusion matrix plot
        plt.figure(figsize=(6, 5))            # Crée une nouvelle figure de taille 6x5 pouces
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Noise (H0)", "Target (H1)"])  # Prépare l'affichage graphique de la matrice de confusion
        disp.plot(cmap=plt.cm.Blues, values_format='d')  # Trace la matrice de confusion avec une palette de bleus et des valeurs entières
        plt.title(f"Frame-Level Confusion Matrix ({split.upper()} Set)")  # Ajoute un titre au graphique
        plt.savefig(f"cfar_frame_confusion_matrix_{split}.png", dpi=300)  # Sauvegarde la figure en PNG haute résolution
        plt.close()                           # Ferme la figure pour libérer la mémoire
        print(f"Saved confusion matrix plot as 'cfar_frame_confusion_matrix_{split}.png'")  # Confirme la sauvegarde

        # ==========================================
        # 5. FRAME-LEVEL ROC PLOT GENERATION (SWAIDAN METHOD)
        # ==========================================
        # Contrairement à la version "cell-by-cell" (qui fait varier le seuil sur les scores
        # SNR de chaque cellule), cette méthode construit la courbe ROC en faisant varier
        # directement la probabilité de fausse alarme (Pfa) du détecteur CFAR, puis en
        # recalculant alpha et les prédictions frame-level à chaque valeur de Pfa.
        print(f"Computing frame-level ROC by sweeping Pfa ({split.upper()})...")  # Indique le début du balayage de Pfa pour ce split
        
        # Collect peak SNR scores (continuous, not binary)
        all_peak_snrs = []                    # Liste des pics de SNR (un score continu par frame, avant seuillage)
        for img_path in image_files:          # Reparcourt toutes les images du split (deuxième passe, dédiée au balayage Pfa)
            base_name = os.path.basename(img_path)         # Récupère le nom de fichier sans le chemin
            file_id = os.path.splitext(base_name)[0]        # Récupère l'identifiant du fichier sans extension
            lbl_path = os.path.join(label_dir, f"{file_id}.txt")  # Construit le chemin du fichier de labels correspondant (non utilisé ici mais cohérent avec la 1ère passe)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Recharge l'image en niveaux de gris
            if img is None:                  # Si la lecture de l'image a échoué
                continue                     # Ignore ce fichier et passe au suivant
                
            img_float = img.astype(np.float32) / 255.0       # Convertit l'image en float et normalise entre 0 et 1
            log_compressed = np.log10(img_float + EPSILON)   # Applique la même compression logarithmique que précédemment
            energy_map = np.square(log_compressed)            # Recalcule la carte d'énergie (identique à la 1ère passe)
            
            kernel_side = 2 * N_T + 1         # Taille du côté du noyau CFAR (identique à process_frame_cells)
            kernel = np.ones((kernel_side, kernel_side), dtype=np.float32)  # Noyau initial rempli de 1
            start_g = N_T - N_G               # Début de la zone de garde dans le noyau
            end_g = N_T + N_G + 1             # Fin de la zone de garde dans le noyau
            kernel[start_g:end_g, start_g:end_g] = 0  # Met à zéro la zone de garde (exclue du calcul de bruit)
            kernel /= N_REF                   # Normalise le noyau par le nombre de cellules de référence
            
            local_noise_mean = cv2.filter2D(energy_map, -1, kernel, borderType=cv2.BORDER_REPLICATE)  # Estime le niveau de bruit local par convolution
            local_noise_mean = np.where(local_noise_mean == 0, 1e-6, local_noise_mean)  # Évite les divisions par zéro
            snr_matrix = energy_map / local_noise_mean  # Calcule la carte de SNR pixel par pixel
            
            valid_interior = np.zeros_like(snr_matrix, dtype=bool)  # Masque booléen pour la zone intérieure valide
            valid_interior[N_T:-N_T, N_T:-N_T] = True  # N'active que la zone évitant les artefacts de bord
            
            peak_snr = np.max(snr_matrix[valid_interior])  # Récupère le pic de SNR de la frame (score continu, indépendant du seuil)
            all_peak_snrs.append(peak_snr)    # Stocke ce score pour le réutiliser à chaque valeur de Pfa testée
        
        # Sweep Pfa values to generate ROC curve
        pfa_sweep = np.logspace(-5, -0.5, 25)  # 25 points from 1e-5 to ~0.3  # Génère 25 valeurs de Pfa réparties logarithmiquement entre 1e-5 et environ 0.316
        fpr_list, tpr_list = [], []            # Listes pour stocker les taux de faux positifs et vrais positifs à chaque valeur de Pfa
        
        for pfa in pfa_sweep:                  # Parcourt chaque valeur de Pfa du balayage
            alpha_sweep, _ = calculate_alpha(N_G, N_T, pfa)  # Recalcule le seuil alpha correspondant à ce Pfa (on ignore n_ref, déjà connu)
            
            # Generate frame predictions at this Pfa
            y_pred_sweep = [1 if snr > alpha_sweep else 0 for snr in all_peak_snrs]  # Reclasse chaque frame en H1/H0 selon le nouveau seuil alpha_sweep
            
            # Compute confusion matrix
            cm_sweep = confusion_matrix(y_frame_true, y_pred_sweep)  # Calcule la matrice de confusion pour ce seuil
            tn, fp, fn, tp = cm_sweep.ravel()  # Extrait les 4 valeurs (vrais négatifs, faux positifs, faux négatifs, vrais positifs)
            
            # Compute FPR and TPR
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calcule le taux de fausses alarmes (FPR) empirique à ce seuil
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Calcule le taux de détection (TPR) empirique à ce seuil
            
            fpr_list.append(fpr)               # Ajoute le FPR calculé à la liste
            tpr_list.append(tpr)               # Ajoute le TPR calculé à la liste
        
        # Compute AUC
        fpr_array = np.array(fpr_list)         # Convertit la liste des FPR en tableau numpy
        tpr_array = np.array(tpr_list)         # Convertit la liste des TPR en tableau numpy
        roc_auc = auc(fpr_array, tpr_array)    # Calcule l'aire sous la courbe ROC (AUC) à partir des points (FPR, TPR) obtenus par balayage de Pfa

        plt.figure(figsize=(8, 6))             # Crée une nouvelle figure de taille 8x6 pouces pour la courbe ROC
        plt.plot(fpr_array, tpr_array, 'o-', color='red', lw=2, label=f'Frame-Level ROC (AUC = {roc_auc:.4f})')  # Trace la courbe ROC (points reliés) avec l'AUC dans la légende
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Trace la diagonale de référence (classifieur aléatoire)
        plt.xlim([0.0, 1.0])                   # Fixe les limites de l'axe des abscisses (FPR) entre 0 et 1
        plt.ylim([0.0, 1.05])                  # Fixe les limites de l'axe des ordonnées (TPR) entre 0 et 1.05
        plt.xlabel('False positive rate')      # Étiquette de l'axe des abscisses
        plt.ylabel('True positive rate')       # Étiquette de l'axe des ordonnées
        plt.title(f'Frame-Level CA-CFAR ROC Curve ({split.upper()} Set)')  # Titre du graphique ROC (méthode frame-level, balayage de Pfa)
        plt.legend(loc="lower right")          # Affiche la légende en bas à droite
        plt.grid(True, linestyle=':', alpha=0.6)  # Ajoute une grille pointillée semi-transparente pour la lisibilité

        plt.savefig(f"cfar_frame_roc_{split}.png", dpi=300)  # Sauvegarde la courbe ROC frame-level en PNG haute résolution
        plt.close()                            # Ferme la figure pour libérer la mémoire
        print(f"Saved frame-level ROC curve as 'cfar_frame_roc_{split}.png'")  # Confirme la sauvegarde de la courbe ROC

if __name__ == "__main__":                     # Point d'entrée du script : ne s'exécute que si le fichier est lancé directement
    main()                                      # Appelle la fonction principale qui orchestre tout le traitement