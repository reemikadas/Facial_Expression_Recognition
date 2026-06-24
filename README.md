# 😊😢 Facial Expression Recognition with Deep Learning

A binary image classification project that uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to detect and classify facial expressions as either **Happy** or **Sad**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)

---

## 🔍 Overview

This project implements a **deep learning pipeline** for facial expression recognition. It covers the complete ML workflow:

1. **Data Preprocessing** — loading, validating, and cleaning image data
2. **Dataset Splitting** — 70% train / 20% validation / 10% test
3. **Model Building** — custom CNN architecture
4. **Training** — 20 epochs with TensorBoard logging
5. **Evaluation** — Precision, Recall, and Binary Accuracy metrics
6. **Prediction** — real-time inference on new images

---

## 📁 Project Structure

```
Facial Expression Recognition/
│
├── data/
│   ├── happy/          # Training images of happy faces (~101 images)
│   └── sad/            # Training images of sad faces
│
├── logs/
│   ├── train/          # TensorBoard training logs
│   └── validation/     # TensorBoard validation logs
│
├── models/
│   └── Sentiment_Image_Classifier_model.h5   # Saved trained model (~42 MB)
│
├── test_img_1.jpg      # Sample test image
├── test_img_2.jpg      # Sample test image
│
└── Facial Expression Recognition with Deep Learning Project.ipynb
```

---

## 🗂️ Dataset

- **Classes:** `Happy` (Class 0) and `Sad` (Class 1)
- **Total images:** ~170 files across 2 classes
- **Image format:** JPEG, JPG, BMP, PNG
- **Input size:** Resized to **256×256×3** (RGB)

> **Data Cleaning:** Dodgy/corrupted images are automatically detected and removed using Python's `imghdr` and `cv2` libraries before training.

**Dataset Split:**
| Split       | Proportion | Batches |
|-------------|------------|---------|
| Training    | 70%        | 4       |
| Validation  | 20%        | 1       |
| Test        | 10%        | 1       |

---

## 🧠 Model Architecture

A custom **Sequential CNN** is built using TensorFlow/Keras:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃ Param #       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 254, 254, 16)   │           448 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 127, 127, 16)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 125, 125, 32)   │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 62, 62, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 60, 60, 16)     │         4,624 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 30, 30, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 14400)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     3,686,656 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           257 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 3,696,625 (14.10 MB)
 Trainable params: 3,696,625 (14.10 MB)
 Non-trainable params: 0 (0.00 B)
```

**Key Details:**
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Output Activation:** Sigmoid
- **Hidden Activation:** ReLU

---

## 📈 Training Results

The model was trained for **20 epochs** with TensorBoard logging:

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1     | 42.97%        | 1.2008     | 37.50%       | 0.8614   |
| 5     | 80.47%        | 0.4660     | 62.50%       | 0.5334   |
| 10    | 89.84%        | 0.2430     | 100.00%      | 0.1361   |
| 15    | 99.22%        | 0.0560     | 100.00%      | 0.0254   |
| 20    | **100.00%**   | **0.0052** | **100.00%**  | **0.0017** |

---

## 📊 Performance Metrics

Evaluated on the **test set** using:

- **Precision**
- **Recall**
- **Binary Accuracy**

> The model achieves **100% validation accuracy** by epoch 9 and maintains it through epoch 20.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/your-username/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
```

### Install Dependencies

```bash
pip install tensorflow opencv-python matplotlib numpy pandas
```

---

## 🚀 Usage

### Run the Notebook

Open the Jupyter Notebook and run all cells:

```bash
jupyter notebook "Facial Expression Recognition with Deep Learning Project.ipynb"
```

### Predict on a New Image

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load saved model
model = load_model('models/Sentiment_Image_Classifier_model.h5')

# Load and preprocess image
img = cv2.imread('your_image.jpg')
resize = tf.image.resize(img, (256, 256))

# Predict
yhat = model.predict(np.expand_dims(resize / 255, 0))

if yhat > 0.5:
    print("Predicted Class: Sad 😢")
else:
    print("Predicted Class: Happy 😊")
```

### View TensorBoard Logs

```bash
tensorboard --logdir logs/
```

---

## 📦 Dependencies

| Library         | Purpose                              |
|----------------|--------------------------------------|
| `tensorflow`    | Deep learning framework              |
| `keras`         | High-level neural network API        |
| `opencv-python` | Image loading and preprocessing      |
| `matplotlib`    | Visualization and plotting           |
| `numpy`         | Numerical computing                  |
| `pandas`        | Data handling                        |
| `imghdr`        | Image format validation              |

---

## 🎯 Results

The trained CNN classifier:
- ✅ Achieves **100% accuracy** on both training and validation sets
- ✅ Correctly identifies **Happy** and **Sad** facial expressions
- ✅ Saved model available at `models/Sentiment_Image_Classifier_model.h5`

**Prediction Example:**

```
Input: test_img_1.jpg
Output: "Predicted Class is Happy" ✅
```

---

## 📝 Notes

- Images are normalized by dividing pixel values by `255` before inference.
- GPU memory growth is configured to prevent Out-Of-Memory errors if a GPU is available.
- Dodgy/unsupported image formats are automatically filtered during data loading.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ using TensorFlow & Keras*
