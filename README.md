# RFDETR-and-YOLO-26-object-detetcions-Implementations
# Object Detection with RF-DETR & YOLO26

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO26](https://img.shields.io/badge/Model-YOLO26-orange)
![RF-DETR](https://img.shields.io/badge/Model-RF--DETR-green)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-0.978-brightgreen)

Real-time detection of **9 object classes** (people, protective equipment and trays) using two modern detectors: **YOLO26** (Ultralytics) and **RF-DETR** (Roboflow). This repository contains the training setup, evaluation results and inference examples for both models.

> **[Describe your use case here in one sentence, e.g. "Monitoring hygiene compliance in a food-production line."]**

---

## Highlights

| Metric | Value |
|---|---|
| Classes | 9 |
| **mAP@0.5 (all classes)** | **0.978** |
| **Best F1 (all classes)** | **0.97** at confidence **0.401** |
| Recall (all classes) | 0.98 |
| Wrong-class errors | only **28 of 20,961** objects (0.13%) |
| Strongest classes (AP@0.5) | pcs 0.994, ungloved 0.993, tray_red 0.992, tray 0.991 |
| Weakest class (AP@0.5) | albag 0.935 (very few examples) |

---

## Classes

| Class | Description | Evaluation instances |
|---|---|---:|
| `person` | A person in the scene | 1,962 |
| `gloved` | Hand wearing a glove | 2,918 |
| `ungloved` | Bare hand | 1,668 |
| `masked` | Face wearing a mask | 68 |
| `unmasked` | Face without a mask | 255 |
| `tray` | Tray | 4,868 |
| `tray_red` | Red tray | 2,456 |
| `pcs` | Pieces on the Board | 6,732 |
| `albag` | Any Red Tray on the Board | 34 |

> Instance counts are derived from the confusion matrix. The dataset is **imbalanced**: `pcs` has about 200 times more examples than `albag`.

---

## Models

| Model | Type | Framework |
|---|---|---|
| **YOLO26** | Real-time one-stage CNN detector | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **RF-DETR** | Real-time transformer (DETR-style) detector | [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) |

---

## Results: YOLO26

### Summary

| Metric | Value |
|---|---|
| mAP@0.5 | **0.978** |
| mAP@0.5:0.95 | **0.932** |
| F1 (all classes) | 0.97 at confidence 0.401 |
| Precision (all classes) | 1.00 at confidence 1.000 |
| Recall (all classes) | 0.98 at confidence 0.000 |
| Overall precision / recall / F1 (from confusion matrix) | 0.970 / 0.985 / 0.977 |

### Per-class performance

AP@0.5 comes from the precision-recall curve. Precision, recall and F1 are calculated from the confusion matrix counts.

| Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| albag | 0.935 | 0.889 | 0.941 | 0.914 | 32 | 4 | 2 |
| gloved | 0.975 | 0.953 | 0.975 | 0.964 | 2,844 | 140 | 74 |
| masked | 0.975 | 0.943 | 0.971 | 0.957 | 66 | 4 | 2 |
| pcs | 0.994 | 0.980 | 0.992 | 0.986 | 6,677 | 137 | 55 |
| person | 0.970 | 0.957 | 0.968 | 0.962 | 1,899 | 86 | 63 |
| tray | 0.991 | 0.972 | 0.987 | 0.980 | 4,805 | 138 | 63 |
| tray_red | 0.992 | 0.972 | 0.986 | 0.979 | 2,421 | 71 | 35 |
| ungloved | 0.993 | 0.964 | 0.990 | 0.977 | 1,652 | 62 | 16 |
| unmasked | 0.980 | 0.981 | 0.988 | 0.984 | 252 | 5 | 3 |
| **All** | **0.978** | **0.970** | **0.985** | **0.977** | **20,648** | **647** | **313** |

### Curves

<p align="center">
  <img src="results/yolo26/BoxPR_curve.png" width="49%" alt="Precision-Recall curve">
  <img src="results/yolo26/BoxF1_curve.png" width="49%" alt="F1-Confidence curve">
</p>
<p align="center">
  <img src="results/yolo26/BoxP_curve.png" width="49%" alt="Precision-Confidence curve">
  <img src="results/yolo26/BoxR_curve.png" width="49%" alt="Recall-Confidence curve">
</p>

- **Precision-Recall:** every class stays near 1.0 precision until very high recall, giving an overall mAP@0.5 of 0.978.
- **F1-Confidence:** the F1 score is almost flat between confidence 0.1 and 0.8, so the model is not sensitive to the exact threshold. It peaks at **0.401**, which is a good default.
- **Precision-Confidence:** precision rises steadily with the threshold and reaches 1.00 at the highest confidence.
- **Recall-Confidence:** recall stays above 0.95 for most classes until about 0.7 confidence, then falls quickly.

### Confusion matrix

<p align="center">
  <img src="results/yolo26/confusion_matrix.png" width="49%" alt="Confusion matrix">
  <img src="results/yolo26/confusion_matrix_normalized.png" width="49%" alt="Normalized confusion matrix">
</p>

What the matrix shows:

- **Very few class mix-ups.** Only 28 of 20,961 objects were assigned to the wrong class.
  - `gloved` and `ungloved`: 10 mix-ups in total (8 gloved predicted as ungloved, 2 the other way).
  - `masked` and `unmasked`: no mix-ups.
  - `tray` and `tray_red`: 13 mix-ups in total (11 `tray_red` predicted as `tray`, 2 the other way).
  - `person` and `tray`: 5 mix-ups in total.
- **Most errors involve the background.**
  - 619 false detections on empty background, mostly `gloved` (138), `pcs` (137) and `tray` (124).
  - 285 missed objects, mostly `gloved` (66), `person` (60), `tray` (59) and `pcs` (55).
- **Per-class recall** (normalized matrix) is 0.97 or higher for every class except `albag` (0.94).

### Limitations

- `albag` has only about 34 evaluation examples, so its metrics are the lowest and the least reliable. More data for `albag`, `masked` and `unmasked` would give more trustworthy numbers.
- The remaining errors are mainly false positives and misses against the background, not confusion between classes. Adding hard negative images (scenes without target objects) is the most direct way to reduce them.

---

## Results: RF-DETR

> **[Add RF-DETR results here: PR curve, F1 curve and confusion matrix, using the same layout as the YOLO26 section.]**

### Model comparison

| Metric | YOLO26 | RF-DETR |
|---|---:|---:|
| mAP@0.5 | 0.978 | [add] |
| mAP@0.5:0.95 | [add] | [add] |
| Best F1 | 0.97 | [add] |
| Inference speed (ms / image) | [add] | [add] |
| Parameters | [add] | [add] |

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install ultralytics rfdetr
```

---

## Dataset

Organize the data in the standard YOLO format, with a `data.yaml` file that lists the 9 classes:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

```yaml
# data.yaml
path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: albag
  1: gloved
  2: masked
  3: pcs
  4: person
  5: tray
  6: tray_red
  7: ungloved
  8: unmasked
```

> Make sure the class order matches the order used during training.

---

## Training

The values below are examples. Replace them with your own settings.

### YOLO26

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")        # choose the size: n / s / m / l / x
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

### RF-DETR

```python
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="dataset",
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
)
```

---

## Evaluation

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="dataset/data.yaml")
print(metrics.box.map50)          # mAP@0.5
```

Validation produces the plots shown above (`BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, and others) in the run folder.

---

## Inference

**Recommended confidence threshold: 0.40** (the peak of the F1 curve).

```python
from ultralytics import YOLO

model = YOLO("weights/yolo26_best.pt")
results = model.predict("path/to/image_or_video.mp4", conf=0.40, save=True)
```

```python
from PIL import Image
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights="weights/rfdetr_best.pth")
detections = model.predict(Image.open("path/to/image.jpg"), threshold=0.40)
```

---

## Repository structure

```
.
├── README.md
├── data.yaml
├── weights/                 # trained models
├── results/
│   ├── yolo26/              # evaluation plots
│   └── rfdetr/
├── notebooks/               # training and evaluation notebooks
└── scripts/                 # inference scripts
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26
- [Roboflow](https://github.com/roboflow/rf-detr) for RF-DETR

## Author

**[Your Name]**: Bio scientist and AI engineer
[YouTube channel](#) · [GitHub](#) · [LinkedIn](#)

## License

**[Choose a license, e.g. MIT]**. Note that Ultralytics models use the AGPL-3.0 license, so check the terms before commercial use.
