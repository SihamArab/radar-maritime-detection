# radar-maritime-detection

Projet de fin d'études de Master, ENSTA Bretagne, laboratoire Lab-STICC, 2026.

Sujet : détection de cibles radar sur des cartes temps-fréquence et classification de formes d'onde, sur le jeu RadDet40K.
Un détecteur classique CA-CFAR 2D est comparé à YOLOv8n et YOLOv10n. La classification des formes d'onde compare ResNet18, 34 et 50 à un CNN dédié.

## Problème

1. **Détection.** Décider si un spectrogramme contient une cible, et où. Référence : CA-CFAR. Détecteurs appris : YOLOv8n et YOLOv10n en mode une seule classe.
2. **Classification.** Identifier la forme d'onde d'une cible parmi 11 classes : Barker, Costas, Frank, P1, P2, P3, P4, Poly, T1, T2, T3.

## Données

RadDet40K, 40 000 spectrogrammes radar simulés, 11 classes de formes d'onde. Le jeu n'est pas inclus dans ce dépôt.
Organisation attendue, au format YOLO. Un fichier d'étiquettes absent ou vide signifie du bruit seul.

```
RADDet/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt     classe cx cy w h, normalisés
```

L'emplacement du jeu se règle avec la variable d'environnement `RADDET_DIR`, par défaut `./RADDet`.

| Partition | Images | Cibles |
|---|---|---|
| train | 20 000 | 10 007 |
| val | 14 000 | 6 983 |
| test | 6 000 | 3 051 |

Les scripts de classification ne gardent que les images contenant une cible. L'indépendance stricte entre les trois partitions n'a pas été vérifiée.

## Organisation du dépôt

```
detection/
  cfar.py              CA-CFAR 2D, décision par image, matrice de confusion et ROC en balayant Pfa
  yolo_benchmark.py    YOLOv8n contre YOLOv10n, détection binaire, évaluation par image et par boîte
classification/
  radar_common.py            chargement des données, normalisation, boucle d'entraînement, métriques
  resnet_benchmark.py        ResNet18, 34 et 50, poids ImageNet, ajustement en deux phases
  custom_cnn_benchmark.py    CNN dédié entraîné de zéro, même chaîne que les ResNet
```

## Méthodes

**CA-CFAR.** L'image est normalisée, passée en échelle logarithmique puis élevée au carré pour obtenir une carte d'énergie.
Le niveau de bruit local est estimé par une moyenne 2D sur une fenêtre de 17 par 17 avec une zone de garde de 5 par 5, soit 264 cellules de référence.
Le score de détection est le rapport entre l'énergie et le bruit local. Le facteur de seuil vaut
`alpha = N_ref * (Pfa^(-1/N_ref) - 1)`, avec une Pfa de conception de 1e-2, soit `alpha` d'environ 4,65.
Une image est déclarée positive si son score maximal dépasse `alpha`.

**YOLO.** Ultralytics, entrée 128 par 128, 50 époques, Adam avec un taux d'apprentissage de 3e-4 et une régularisation de 1e-4, patience 10, une seule classe fusionnée.
Les augmentations sont désactivées, sauf une translation de 0,05.
L'évaluation se fait par image : une image est positive si au moins une boîte dépasse le seuil de confiance. Cela permet de calculer une vraie Pfa et de comparer avec le CA-CFAR.

**ResNet.** Préentraînés sur ImageNet. Première convolution avec un pas de 1, max-pooling retiré.
Tête : Dropout 0,4, Linear 512, ReLU, BatchNorm, Dropout 0,3, Linear 11.
Phase 1 : entraînement de la tête sur 5 époques. Phase 2 : ajustement de tout le réseau sur 40 époques, avec des taux d'apprentissage par couche, AdamW, recuit cosinus et une perte pondérée par classe.

**CNN dédié.** Quatre blocs Conv 3 par 3, BatchNorm, ReLU, MaxPool 2 par 2, avec 32, 64, 128 puis 256 canaux, puis un pooling moyen adaptatif et la même tête que les ResNet.
Entraîné de zéro avec le même optimiseur, la même perte et le même nombre total d'époques.

Tous les classifieurs partagent `radar_common.py` : même partition, même normalisation, mêmes métriques.
Le choix du modèle se fait sur la perte de validation, et le jeu de test n'est évalué qu'une fois à la fin.

## Utilisation

```bash
pip install -r requirements.txt
export RADDET_DIR=/chemin/vers/RADDet

python detection/cfar.py
python detection/yolo_benchmark.py
cd classification
python resnet_benchmark.py
python custom_cnn_benchmark.py
```

## Licence

MIT
