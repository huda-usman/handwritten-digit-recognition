# ⚡ DIGITAI — Handwritten Digit Recognition

A full-stack deep learning web app that classifies handwritten digits in real time using a custom CNN trained on MNIST.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.55%25-brightgreen)]()

---

## 📸 Screenshots

| Home Page | Studio — Drawing | Studio — Upload |
|-----------|-----------------|-----------------|
| ![Home](HomePage.png) | ![Canvas](Canvas.png) | ![Studio](Studio.png) |

---

## ✨ Features

- **Draw Mode** — Freehand canvas drawing with adjustable brush size
- **Upload Mode** — Upload any PNG/JPG digit image for instant classification
- **Real-time Results** — Predicted digit, confidence score, and full class probabilities
- **2-Page App** — Beautiful Home page + Studio with 3-column layout
- **~99.55% Accuracy** — Custom CNN-v2 trained with augmentation on 70k MNIST samples

---

## 🧠 Model Architecture

```
Input (28×28×1)
  → Conv2D(32) → Conv2D(32) → MaxPool → Dropout(0.25)
  → Conv2D(64) → Conv2D(64) → MaxPool → Dropout(0.25)
  → Flatten → Dense(256) → Dropout(0.5)
  → Dense(10, softmax)
```

- **Optimizer:** Adam (lr=1e-3)
- **Augmentation:** rotation ±10°, zoom ±10%, shifts ±10%
- **Callbacks:** ReduceLROnPlateau + EarlyStopping
- **Test Accuracy:** 99.55% · Inference: <50ms

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/huda-usman/digitai.git
cd digitai
```

### 2. Install dependencies
```bash
pip install streamlit streamlit-drawable-canvas tensorflow opencv-python pillow numpy
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** The trained model (`mnist_cnn_v2_model.keras`) is included — no training needed!

---

## 📁 Project Structure

```
digitai/
├── app.py                      # Main Streamlit app (Home + Studio pages)
├── mnist_cnn_v2_model.keras    # Pre-trained CNN model
├── train_on_colab.ipynb        # Google Colab GPU training notebook
├── Digits/                     # Sample test digit images
│   ├── 0.jpeg
│   ├── 1.jpeg
│   └── ...
├── HomePage.png                # Screenshot — Home page
├── Studio.png                  # Screenshot — Studio page
├── Canvas.png                  # Screenshot — Drawing canvas
├── Canvas2.png
├── Canvas3.png
└── README.md
```

---

## 🔬 Training (Optional)

The model is already trained and included. If you want to retrain:

**Option A — Google Colab (recommended, free GPU):**
1. Open `train_on_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set Runtime → GPU
3. Run all cells — training takes ~5 min on T4 GPU
4. Download the saved `mnist_cnn_v2_model.keras` and replace the existing one

**Option B — Local:**
```bash
python -c "
import tensorflow as tf, numpy as np
# See train_on_colab.ipynb for full training code
"
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| Drawing | streamlit-drawable-canvas |
| ML Framework | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow |
| Dataset | MNIST (70k samples) |
| Training | Google Colab (T4 GPU) |

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
streamlit-drawable-canvas>=0.9.3
tensorflow>=2.13.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

---

## 👤 Author

**Huda** — [@huda-usman](https://github.com/huda-usman)

---

## 📄 License

MIT License — feel free to use, modify, and share.
