# BreastInsight — CNN for Breast Ultrasound Image Classification

A Convolutional Neural Network that classifies breast ultrasound images into three categories: **normal, benign, and malignant**. Trained on the BUSI dataset, reaching **69% validation accuracy** (macro-avg recall 51%) after fixing two bugs in the original pipeline — see [Data Integrity Fix](#data-integrity-fix-masks-were-being-trained-on) below.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Problem.** Manual review of breast ultrasound images is time-consuming and requires specialist expertise. A well-tuned image classifier can act as a screening aid, flagging suspicious images for radiologist review.

**Approach.**

1. Loaded the BUSI (Breast Ultrasound Images) dataset — three classes: normal, benign, malignant — **excluding the segmentation masks** (see below).
2. Applied image preprocessing — resizing, normalization, and data augmentation to reduce overfitting on the small dataset.
3. Trained a 3-layer Convolutional Neural Network with dropout and evaluated on a held-out validation set.
4. Evaluated with accuracy, confusion matrix, and per-class recall (recall matters most for the malignant class).

**Result.** **69% validation accuracy** on 3-class classification. Per-class recall: benign 0.91, malignant 0.47, normal 0.14 — the model is reasonably reliable on benign cases but misses roughly half of malignant cases and struggles badly on normal scans (see [Limitations](#limitations)).

## Data Integrity Fix: masks were being trained on

The original notebook pointed `image_dataset_from_directory` directly at `Dataset_BUSI_with_GT/`, which holds each ultrasound scan **alongside its segmentation mask** (`*_mask.png`). Globbing that folder pulls in all **1578** files — 780 real ultrasound images plus **798 binary masks** — and the model was trained and evaluated on that mixed set as if every file were a genuine 3-class ultrasound image. Masks carry none of the texture information the classifier actually needs, so this corrupted both training and every accuracy figure the notebook reported, including the headline "87% test accuracy" claim, which does not correspond to any number the notebook actually produced.

A second, independent bug compounded this: the classification-report cell built `true_labels` and `predicted_labels` from **two separate iterations** over the shuffled validation `tf.data.Dataset`. Each iteration reshuffles, so the two lists were drawn in different orders and no longer corresponded to the same images — which is why the old notebook showed a nonsensical 46% classification-report accuracy against a 70% `model.evaluate()` accuracy on the very same data.

Both are fixed in the current notebook: a `clean_dataset/` copy is built with masks filtered out before training, and true/predicted labels are now collected in a single aligned pass. The numbers above are the real, verified result of that fix.

## Tech Stack

- **Python**, **Jupyter Notebook**
- **TensorFlow / Keras** — CNN model, ImageDataGenerator, callbacks
- **NumPy**, **Matplotlib** — array handling and evaluation plots
- **PIL / OpenCV** — image I/O and preprocessing

## Dataset

Source: [Breast Ultrasound Images Dataset (BUSI) — Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

- 780 images across 3 classes: normal, benign, malignant
- Grayscale ultrasound scans with ground-truth masks

## Model Architecture

3-layer CNN — Conv2D → MaxPool blocks with ReLU, dropout regularization, and a dense linear output for 3-way classification. Trained with the Adam optimizer and sparse categorical cross-entropy loss (`from_logits=True`). The first layer is a `Rescaling(1./255)` layer, so the model expects raw 0–255 pixel input — do not pre-normalize images before calling `predict`.

## Project Structure

```
BreastInsight/
├── BreastInsight.ipynb    # End-to-end notebook: preprocessing, training, evaluation
├── BreastInsight.h5       # Trained Keras model weights
├── model.fix/             # Alternate model checkpoints
├── LICENSE
└── README.md
```

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/aljuhaeda/BreastInsight.git
cd BreastInsight
```

**2. Install dependencies**

```bash
pip install tensorflow keras numpy matplotlib pillow jupyter
```

**3. Download the dataset**

Download the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place it in `Dataset_BUSI_with_GT/` at the project root (masks and all — the notebook filters them out for you).

**4. Open the notebook**

```bash
jupyter notebook BreastInsight.ipynb
```

Run all cells to reproduce preprocessing, training, and evaluation. The early cells build a `clean_dataset/` copy with `*_mask.png` files excluded — training happens on that copy, not on `Dataset_BUSI_with_GT/` directly.

## Inference on a New Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("BreastInsight.h5")
img = image.load_img("your_image.png", target_size=(224, 224))
# Do NOT divide by 255 here — the model's first layer already rescales 0-255 input.
x = np.expand_dims(image.img_to_array(img), axis=0)
pred = model.predict(x)
class_names = ["benign", "malignant", "normal"]
print(class_names[np.argmax(pred)])
```

## Limitations

- **Small, imbalanced dataset (780 images: 437 benign / 210 malignant / 133 normal)** — augmentation helps but generalization to different imaging equipment or populations is not guaranteed. The class imbalance shows up directly in the results: normal-class recall is only 0.14, meaning the model rarely identifies a normal scan correctly.
- **Malignant recall (0.47) is not screening-grade.** The model misses over half of malignant cases in validation — nowhere near sufficient for any real triage use.
- **Not for clinical use** — this is a research/educational project. Any real-world screening decision requires validation on much larger, multi-site datasets and regulatory approval.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
