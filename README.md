# radar-maritime-detection
Master's end-of-studies project, ENSTA Bretagne, 2026.

Subject: Radar target detection on Range-Doppler maps and waveform classification, on the RadDet40K dataset.
A classical 2D CA-CFAR detector is compared with YOLOv8n and YOLOv10n, and waveform classification is compared between ResNet18/34/50 and a dedicated CNN.

## Problem

1. **Detection.** Decide whether a spectrogram contains a target, and where. Baseline: CA-CFAR. Learned detectors: YOLOv8n and YOLOv10n in single-class mode.
2. **Classification.** Identify the waveform of a detected target among 11 classes: Barker, Costas, Frank, P1, P2, P3, P4, Poly, T1, T2, T3.

## Data

RadDet40K, 40 000 simulated radar spectrograms, 11 waveform classes. The dataset is not included in this repository.
Expected layout, in YOLO format, where a missing or empty label file means noise only:

```
RADDet/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt     class cx cy w h, normalised
```

Set the dataset location with the environment variable `RADDET_DIR`, default `./RADDet`.

| Split | Images | Targets |
|---|---|---|
| train | 20 000 | 10 007 |
| val | 14 000 | 6 983 |
| test | 6 000 | 3 051 |

The classification scripts keep only images that contain a target. Strict independence between the three splits has not been verified yet.

## Repository layout

```
detection/
  cfar.py              2D CA-CFAR, frame-level decision, confusion matrix and ROC by sweeping Pfa
  yolo_benchmark.py    YOLOv8n vs YOLOv10n, binary detection, image-level and box-level evaluation
classification/
  radar_common.py            shared data loading, normalisation, training loop, metrics
  resnet_benchmark.py        ResNet18/34/50, ImageNet weights, two-phase fine-tuning
  custom_cnn_benchmark.py    dedicated CNN trained from scratch, same pipeline as the ResNets
```

## Methods

**CA-CFAR.** The image is normalised, log-compressed and squared into an energy map. The local noise level is
estimated by a 2D mean over a 17x17 window with a 5x5 guard zone, which gives 264 reference cells. The detection
score is the ratio between the energy and the local noise. The threshold factor is
`alpha = N_ref * (Pfa^(-1/N_ref) - 1)`, with a design Pfa of 1e-2, so `alpha` is about 4.65. A frame is declared
positive if the peak score exceeds `alpha`.

**YOLO.** Ultralytics, 128x128 input, 50 epochs, Adam with learning rate 3e-4 and weight decay 1e-4, patience 10, one merged class.
Augmentations are disabled except a 0.05 translation. Evaluation is image-level: an image is positive if at least
one box exceeds the confidence threshold, which allows a true Pfa to be computed and compared with CA-CFAR.

**ResNet.** Pretrained on ImageNet. First convolution at stride 1, max-pooling removed. Head: Dropout 0.4, Linear 512,
ReLU, BatchNorm, Dropout 0.3, Linear 11. Phase 1 trains the head for 5 epochs, phase 2 fine-tunes the whole network for 40 epochs
with per-layer learning rates, AdamW, cosine annealing and a class-weighted loss.

**Dedicated CNN.** Four blocks Conv 3x3, BatchNorm, ReLU, MaxPool 2x2 with 32, 64, 128 and 256 channels, then adaptive average pooling
and the same head as the ResNets. Trained from scratch with the same optimiser, loss and total epoch budget.

All classifiers share `radar_common.py`: same split, same normalisation, same metrics. Model selection uses the
validation loss, and the test set is evaluated once at the end.

## Usage

```bash
pip install -r requirements.txt
export RADDET_DIR=/path/to/RADDet

python detection/cfar.py
python detection/yolo_benchmark.py
cd classification
python resnet_benchmark.py
python custom_cnn_benchmark.py
```

## Known limitations

- The images are false-colour PNG files. `cfar.py` reads them as grayscale, so the detector works on luminance, which is not the signal power.
  The energy map also reverses the intensity order, so dark pixels receive high energy. This needs to be checked against the colour map used.
- The classification pipeline applies Otsu background suppression by default, and an image with several targets takes its majority class.
- The ROC curve of `cfar.py` sweeps Pfa from 1e-5 to about 0.3 only, so the reported AUC covers that range.
- `yolo_benchmark.py` compares YOLO with reference CA-CFAR counts taken from an earlier report by M. Swaidan, not with the output of `cfar.py`.
- The CA-CFAR configuration here differs from the 10-reference-cell configuration of that report.
- No results are reported in this repository.

## Acknowledgements

Project carried out at ENSTA Bretagne under the supervision of Abdelmalek Toumi and Jean-Christophe Cexus.
The baseline CA-CFAR reference figures come from the report of Mohamad Swaidan.

## License

MIT
