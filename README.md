# 🧠 Breast Cancer Detection using Deep Learning (Neural Networks)

This project applies a simple but effective **deep learning neural network** to classify breast cancer tumors as **malignant** or **benign** using the well-known **Wisconsin Breast Cancer Dataset**.

The goal is to demonstrate how deep learning can be applied to structured (tabular) medical data for binary classification tasks, using **TensorFlow/Keras**.

---

## 🧬 Dataset

We use the **Breast Cancer Wisconsin Diagnostic Dataset**, directly accessible from `sklearn.datasets`.

- **Samples:** 569
- **Features:** 30 numeric features per sample
- **Labels:** 
  - `0` = Malignant
  - `1` = Benign

## 🧠 Deep Learning Model

A fully connected **feedforward neural network (Multilayer Perceptron)** is built using `tensorflow.keras`.

### 🔧 Architecture

- **Input Layer:** 30 neurons (for each feature)
- **Hidden Layer:** 20 neurons, with **ReLU** activation
- **Output Layer:** 2 neurons, with **Sigmoid** activation (for binary classification)

### 🧪 Training

- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam`
- **Epochs:** 10
- **Validation Split:** 10%

### 📈 Training Results

The model reaches **high accuracy** on both the training and validation sets.

The following plots are generated:

- **Training vs Validation Accuracy**
- **Training vs Validation Loss**

These plots help visualize how well the model is learning and generalizing.

### 🔍 Prediction

Once trained, the model can predict whether a given sample is **malignant** or **benign**, based on its 30 input features.

```python
# Example:
input_data = (20.6, 29.33, 140.1, 1265, ..., 0.124)

# Prediction output:
# ➤ The breast cancer is Benign
