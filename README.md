# ☀️ SolarGuard AI

### Image-Based Solar Cell Defect Inspection System

SolarGuard AI is an image-based solar cell inspection prototype that uses **electroluminescence (EL) images** and a **MobileNetV2 transfer-learning model** to classify solar cell images into four defect-probability assessment classes.

The project provides a simple **Streamlit web interface** where users can upload an EL image and receive a model prediction along with class probabilities.

---

## 🔍 Features

* Upload a solar cell EL image
* Automatic image preprocessing
* MobileNetV2-based image classification
* Four defect-probability assessment classes
* Prediction confidence
* Class probability visualization
* Simple and interactive Streamlit interface
* Single-image inspection workflow

---

## 🧠 Model

SolarGuard AI uses **MobileNetV2** with transfer learning.

### Model pipeline

```text
EL Image
   ↓
Grayscale Conversion
   ↓
Resize to 224 × 224
   ↓
Grayscale → RGB
   ↓
MobileNetV2 Preprocessing
   ↓
MobileNetV2
   ↓
Classification Head
   ↓
4-Class Prediction
```

The model was initially trained with the ImageNet-pretrained MobileNetV2 backbone, followed by data augmentation and fine-tuning of the final layers.

---

## 📊 Dataset

The project uses the **ELPV (Electroluminescence Photovoltaic) Dataset**.

The dataset contains **2,624 electroluminescence images** of photovoltaic cells.

The original images are **300 × 300 grayscale images**.

For model input, the images are resized to **224 × 224** and converted to three channels to match the MobileNetV2 input format.

### Classes

| Class                      | Description                            |
| -------------------------- | -------------------------------------- |
| Normal                     | Normal defect-probability assessment   |
| Low Defect Confidence      | Low defect-probability assessment      |
| Moderate Defect Confidence | Moderate defect-probability assessment |
| High Defect Confidence     | High defect-probability assessment     |

> These classes represent the dataset's defect-probability assessments and should not be interpreted as direct physical defect-severity measurements.

---

## 📈 Model Performance

The final model was evaluated on a held-out test set.

**Test Accuracy: 62.18%**

The model performed better on the Normal class and showed more difficulty distinguishing the Low and Moderate defect-probability classes.

This reflects the class imbalance and difficulty of the dataset and is an important limitation of the current prototype.

---

## 🖥️ Streamlit Application

The application provides:

### 1. Image Upload

Users can upload a `.jpg`, `.jpeg`, or `.png` EL image.

### 2. Inspection Result

The application displays:

* Predicted class
* Model confidence

### 3. Class Probabilities

The probability assigned to each of the four classes is displayed using progress bars.

### 4. Model Processing

The application shows the processing configuration:

* Input type: EL Image
* Original image: 300 × 300
* Model input: 224 × 224
* Channels: Grayscale → RGB

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Pillow
* Streamlit
* ELPV Dataset

---

## 📁 Project Structure

```text
SolarGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    └── SolarGuard_AI_Final.keras
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SolarGuard-AI
```

### 2. Install dependencies

It is recommended to use **Python 3.12**.

```bash
py -3.12 -m pip install -r requirements.txt
```

### 3. Run the application

```bash
py -3.12 -m streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🚀 How to Use

1. Open the SolarGuard AI application.
2. Upload a solar cell EL image.
3. Wait for the model to process the image.
4. View the predicted defect-probability class.
5. Check the model confidence.
6. Review the probability distribution across all four classes.

---

## ⚠️ Model Limitations

SolarGuard AI is currently a **research and demonstration prototype**.

The model was trained and evaluated using the ELPV dataset, so performance may vary when images come from different sources, imaging systems, or operating conditions.

The current test accuracy is **62.18%**, and the model has difficulty distinguishing some of the lower-confidence defect classes.

Therefore, the predictions should **not be treated as a replacement for professional or industrial solar-cell inspection**.

---

## 🎯 Future Improvements

Possible future improvements include:

* Training with larger and more diverse EL datasets
* Improving performance on minority classes
* Testing additional CNN architectures
* More extensive hyperparameter tuning
* Evaluation on external datasets
* Batch image inspection
* Deployment as a production-ready inspection system

---

## 👩‍💻 Author

**Padma Lochini**

BTech Student | AIML / Technology Enthusiast

---

## 📌 Project Status

**Status:** Working Prototype

SolarGuard AI demonstrates how transfer learning and computer vision can be applied to electroluminescence-based solar cell inspection.
