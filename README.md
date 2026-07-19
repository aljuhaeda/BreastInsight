# BreastInsight — CNN for Breast Ultrasound Image Classification

A Convolutional Neural Network that classifies breast ultrasound images into three categories: **normal, benign, and malignant**. Trained on the BUSI dataset with **87% test accuracy**.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Problem.** Manual review of breast ultrasound images is time-consuming and requires specialist expertise. A well-tuned image classifier can act as a screening aid, flagging suspicious images for radiologist review.

**Approach.**

1. Loaded the BUSI (Breast Ultrasound Images) dataset — three classes: normal, benign, malignant.
2. Applied image preprocessing — resizing, normalization, and data augmentation to reduce overfitting on the small dataset.
3. Trained a 3-layer Convolutional Neural Network with dropout and evaluated on a held-out test set.
4. Evaluated with accuracy, confusion matrix, and per-class recall (recall matters most for malignant class).

**Result.** **87% test accuracy** on 3-class classification.

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

3-layer CNN — Conv2D → MaxPool blocks with ReLU, dropout regularization, and a dense softmax output for 3-way classification. Trained with Adam optimizer and categorical cross-entropy loss.

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

Download the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place it in `Dataset_BUSI_with_GT/` at the project root.

**4. Open the notebook**

```bash
jupyter notebook BreastInsight.ipynb
```

Run all cells to reproduce preprocessing, training, and evaluation.

## Inference on a New Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("BreastInsight.h5")
img = image.load_img("your_image.png", target_size=(224, 224))
x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
pred = model.predict(x)
class_names = ["benign", "malignant", "normal"]
print(class_names[np.argmax(pred)])
```

## Limitations

- **Small dataset (780 images)** — augmentation helps but generalization to different imaging equipment or populations is not guaranteed.
- **Not for clinical use** — this is a research/educational project. Any real-world screening decision requires validation on much larger, multi-site datasets and regulatory approval.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
