# Classification d'images radar SAR, MSTAR-10

ARAB Siham et OURZIK Koceila, Master Signal et Télécommunication, UBO, 2025-2026.
Encadrant : M. TOUMI Abdelmalek.

Projet intégrateur de Master. Il s'agit de reconnaître dix types de véhicules militaires sur des images radar à synthèse d'ouverture, à partir du jeu de données MSTAR-10.
Un CNN léger conçu pour le projet est comparé à quatre réseaux préentraînés : AlexNet, VGG-16, ResNet-50 et DenseNet-121.

## Données

Le jeu MSTAR-10 n'est pas inclus dans ce dépôt. Il est disponible sur https://www.sdms.afrl.af.mil/index.php?collection=mstar.

- Entraînement : prises de vue à 17 degrés de dépression.
- Test : prises de vue à 15 degrés de dépression.
- Une image par fichier png ou jpg, rangée dans un sous-dossier par classe.

## Structure attendue

```
mstar_sar_classification.py
requirements.txt
mstar_data/
    train/
        classe1/ ... classe10/
    test/
        classe1/ ... classe10/
```

Les sous-dossiers de classes doivent s'appeler `classe1` à `classe10`. Ces noms sont définis dans `Config.CLASS_NAMES`.

## Méthode

**Prétraitement.** Chaque image est passée en niveaux de gris, redimensionnée à 128 par 128, puis convertie en échelle logarithmique avec log de 1 plus l'intensité, et normalisée entre 0 et 1.

**Augmentation, sur l'entraînement uniquement.** Rotations de 10 degrés au plus, translations jusqu'à 10 pour cent, et bruit de speckle multiplicatif simulé, de variance 0,05.

**CNN léger.** Trois blocs convolutifs de 3 par 3 avec 32, 64 puis 128 filtres, chacun suivi d'un max pooling. Puis un dropout à 0,5, une couche dense de 512 neurones, un second dropout et 10 sorties. Environ 13 millions de paramètres.

**Réseaux préentraînés.** Poids ImageNet. La première couche convolutive est remplacée pour accepter une image à un canal, elle repart donc de zéro. La dernière couche est remplacée par 10 sorties.

**Entraînement.** Adam, taux d'apprentissage 1e-4, régularisation L2 de 1e-5, lots de 32 images, graine 42. Arrêt anticipé après 15 époques sans amélioration. Le CNN léger s'entraîne sur 100 époques au maximum, les réseaux préentraînés sur 30 au maximum.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Les paramètres sont regroupés dans la classe `Config` du script, par exemple :

```python
DATA_ROOT = Path('./mstar_data')
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
PATIENCE = 15
USE_AUGMENTATION = True
PRETRAINED_MODELS = {'AlexNet': True, 'VGG16': True, 'ResNet50': True, 'DenseNet121': True}
```

Pour entraîner un seul réseau préentraîné, mettre les autres à `False` dans `PRETRAINED_MODELS`.

## Utilisation

```bash
python mstar_sar_classification.py
```

Le script enchaîne le chargement des données, le CNN léger, puis les quatre réseaux préentraînés. L'ensemble prend environ 12 heures sur CPU.
Les sorties sont écrites dans `outputs/` : modèles dans `models/`, figures dans `plots/`, tableau comparatif et rapports dans `results/`.

## Limites

- Le jeu de test sert aussi à choisir le meilleur modèle et à déclencher l'arrêt anticipé. Il n'y a pas de jeu de validation séparé, les résultats sont donc légèrement optimistes.
- Aucun résultat n'est reporté dans ce dépôt.
- Le jeu de données n'est pas fourni et le script n'a pas été testé sans lui.
