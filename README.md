# 🦴 Bone Fracture Detection using Deep Learning (ResNet18)

## 📌 Overview

This project implements a deep learning-based system for automatic bone fracture detection from X-ray images.

A transfer learning approach is applied using a pre-trained ResNet-18 model adapted for grayscale medical imaging. The system classifies X-ray images into:

- Fractured
- Not Fractured

The project demonstrates how deep learning can assist medical image analysis by providing accurate and reliable binary classification.

---

# 🎯 Problem Statement

Manual X-ray analysis requires medical expertise and can be time-consuming.  
Automated fracture detection systems can:

- Assist radiologists
- Reduce diagnostic workload
- Improve early detection
- Provide decision support in clinical environments

This project aims to build a robust binary classifier for fracture detection using transfer learning and medical image enhancement techniques.

---

# 🧠 Model Architecture

## Backbone Network
- ResNet-18 (Pre-trained on ImageNet)

## Key Modifications
- First convolution layer modified for **grayscale input**
- Final fully connected layer replaced with **single output neuron**
- Sigmoid activation used during inference
- Binary classification using BCEWithLogitsLoss

Transfer learning allows leveraging rich feature representations learned from large-scale datasets.

---

# 🖼 Image Preprocessing

Medical X-ray images often suffer from low contrast.

To enhance fracture visibility:

### ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE improves:
- Edge visibility
- Bone structure clarity
- Fracture pattern detection
- Noise control

This significantly improves model performance in medical image tasks.

---

# 🔄 Data Augmentation (Training Only)

To improve generalization and reduce overfitting:

- Random rotation
- Random resized cropping
- Horizontal flipping
- Brightness & contrast jittering

Validation and test sets remain unchanged for fair evaluation.

---

# ⚙ Training Configuration

- Image Size: 224 × 224
- Batch Size: 32
- Learning Rate: 1e-4
- Epochs: 20
- Optimizer: Adam
- Loss Function: BCEWithLogitsLoss
- Early Stopping (Patience = 5)

Best model is automatically saved as:

```
best_model.pth
```

---

# 📈 Model Performance

## ✅ Test Accuracy: ~97%

### Classification Report:
- High Precision
- High Recall
- Strong F1-score balance

### ROC Curve:
- AUC ≈ 0.99

This indicates excellent discrimination between fractured and non-fractured cases.

---

# 📊 Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC Score

These metrics provide a comprehensive assessment, especially important in medical applications.

---

# 💾 Model Saving & Deployment

The trained model is saved as:

```
best_model.pth
```

This allows:

- Reloading without retraining
- Deployment in desktop or web applications
- Integration into medical AI systems

---

# 🔍 Single Image Prediction

The system includes an inference function that:

- Applies identical preprocessing (including CLAHE)
- Uses configurable decision threshold
- Outputs:
  - Predicted class
  - Confidence score

Example Output:
```
Predicted Class: fractured
Confidence: 0.99
```

---

# 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 🧩 Key Technical Highlights

- Transfer Learning with ResNet18
- Medical image enhancement using CLAHE
- Custom PyTorch Dataset class
- Binary classification with BCEWithLogitsLoss
- Early stopping implementation
- ROC & AUC evaluation
- Model checkpoint saving
- Single image inference pipeline

---

# 🚀 Future Improvements

- Fine-tune deeper ResNet layers
- Add Grad-CAM visualization for explainability
- Deploy as FastAPI medical inference service
- Convert to ONNX for optimized inference
- Implement uncertainty estimation

---

# 👨‍💻 Author

Samir Elhadad  
AI & Data Science Student  
Deep Learning & Computer Vision Enthusiast  

---

## 📌 Project Type

Medical AI – Computer Vision – Deep Learning – Transfer Learning – Binary Classification
