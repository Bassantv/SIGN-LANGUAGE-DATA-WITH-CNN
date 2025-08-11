# Sign Language Recognition with CNN

This project builds and trains a **Convolutional Neural Network (CNN)** to recognize American Sign Language (ASL) letters using the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

---

## 📂 Dataset
- **Training Data:** `sign_mnist_train.csv`
- **Testing Data:** `sign_mnist_test.csv`
- Each image is **28×28 grayscale**, stored as flattened pixel values.
- Labels represent **24 letters** (A–Y excluding J and Z, which require motion).

---

## 🛠 Preprocessing
- Normalize pixel values to the range `[0, 1]`.
- Reshape to `(28, 28, 1)` for CNN input.
- One-hot encode labels (`num_classes=24`).
- **Data Augmentation** with `ImageDataGenerator`:
  - `rotation_range=10`
  - `width_shift_range=0.1`
  - `height_shift_range=0.1`
  - `zoom_range=0.1`

---



**Loss:** `categorical_crossentropy`  
**Optimizer:** `Adam`  
**Metrics:** Accuracy  

---

## 🚀 Training
- **Batch size:** 32
- **Epochs:** 20
- Training data augmented with `ImageDataGenerator`
- Validation data from held-out portion of training set

---

## 📊 Results
- Achieved **>90% validation accuracy** (exact value may vary depending on run).
- Model successfully predicts ASL letters from unseen images.

## 🧠 Model Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
])