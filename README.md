# Breast Ultrasound Classification Dashboard

A deep learning project for classifying breast ultrasound images into three categories: **Normal**, **Benign**, and **Malignant** using two state-of-the-art CNN architectures: **VGG16** and **ResNet50**.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Results](#results)

## 🎯 Overview

This project implements a comprehensive deep learning pipeline for breast ultrasound image classification. It compares two popular CNN architectures (VGG16 and ResNet50) using transfer learning, fine-tuning, and provides an interactive dashboard for real-time predictions with Grad-CAM visualization.

### Dataset

The project uses the BUSI (Breast Ultrasound Images) dataset with ground truth masks. The dataset is organized into three folders:
- `normal/` - Normal breast tissue images
- `benign/` - Benign tumor images  
- `malignant/` - Malignant tumor images

**Dataset Path:**
```
D:\Karishma(ECE)\Dataset_BUSI_with_GT\
├── normal\
├── benign\
└── malignant\
```

## ✨ Features

- **Two CNN Models**: VGG16 and ResNet50 with transfer learning
- **Two-Phase Training**: Feature extraction followed by fine-tuning
- **Data Augmentation**: Random flips, brightness, contrast, and crop
- **Grad-CAM Visualization**: Visual explanation of model predictions
- **Interactive Dashboard**: Streamlit web interface with multiple views
- **Model Comparison**: Side-by-side comparison of both models
- **Performance Metrics**: Confusion matrices, ROC curves, and classification reports
- **Best Model Selection**: Automatic determination of the best performing model

## 📁 Project Structure

```
project/
├── data_utils.py          # Dataset loading and preprocessing
├── models.py              # Model architecture definitions
├── train_vgg16.py         # VGG16 training script
├── train_resnet50.py      # ResNet50 training script
├── evaluate_models.py     # Model evaluation and metrics
├── gradcam_utils.py       # Grad-CAM visualization utilities
├── app.py                 # Streamlit dashboard
├── models/                # Trained model files (.h5)
│   ├── vgg16_final.h5
│   └── resnet50_final.h5
├── figs/                  # Evaluation figures
│   ├── vgg16_cm.png
│   ├── vgg16_roc.png
│   ├── resnet50_cm.png
│   └── resnet50_roc.png
├── README.md              # This file
└── QUICKSTART.md          # Quick start guide
```

## 📦 Requirements

### Python Version
- Python 3.8 or higher

### Required Packages

Install all dependencies using:

```bash
pip install tensorflow streamlit pillow numpy scikit-learn matplotlib seaborn opencv-python
```

Or install from requirements file (if provided):

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web dashboard framework
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical computations
- **scikit-learn**: Evaluation metrics
- **Matplotlib/Seaborn**: Visualization
- **OpenCV**: Image processing for Grad-CAM

## 🚀 Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install tensorflow streamlit pillow numpy scikit-learn matplotlib seaborn opencv-python
   ```

3. **Verify dataset path**:
   - Ensure your dataset is located at: `D:\Karishma(ECE)\Dataset_BUSI_with_GT\`
   - The dataset should have three subfolders: `normal/`, `benign/`, `malignant/`

4. **Create necessary directories**:
   ```bash
   mkdir models figs
   ```

## ⚡ Quick Start

For a detailed quick start guide, see [QUICKSTART.md](QUICKSTART.md).

**TL;DR:**
1. Train models: `python train_vgg16.py` and `python train_resnet50.py`
2. Evaluate: `python evaluate_models.py`
3. Run dashboard: `streamlit run app.py`

## 📖 Usage

### 1. Model Training

#### Train VGG16 Model
```bash
python train_vgg16.py
```

This script:
- Loads and preprocesses the dataset
- Splits data into train/validation/test (70%/15%/15%)
- Trains VGG16 in two phases:
  - **Phase 1**: Feature extraction (base frozen, ~20-30 epochs)
  - **Phase 2**: Fine-tuning (last conv block unfrozen, ~10 epochs)
- Saves the final model to `models/vgg16_final.h5`

#### Train ResNet50 Model
```bash
python train_resnet50.py
```

Similar process for ResNet50 model, saved to `models/resnet50_final.h5`

### 2. Model Evaluation

```bash
python evaluate_models.py
```

This generates:
- Classification reports (precision, recall, F1-score)
- Confusion matrices saved to `figs/vgg16_cm.png` and `figs/resnet50_cm.png`
- ROC curves saved to `figs/vgg16_roc.png` and `figs/resnet50_roc.png`

### 3. Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Dashboard Features:

1. **📊 Comparison View**: Side-by-side comparison of both models
2. **🔵 VGG16 View**: Detailed VGG16 predictions and visualizations
3. **🔴 ResNet50 View**: Detailed ResNet50 predictions and visualizations
4. **🏆 Best Model View**: Automatic comparison showing which model performs better

#### Using the Dashboard:

1. Upload an ultrasound image (PNG, JPG, JPEG)
2. Select a view from the sidebar
3. View predictions, Grad-CAM heatmaps, probabilities, and performance metrics
4. Check the "Best Model" section to see which model has higher confidence

## 🔬 Model Training Details

### Architecture

Both models use transfer learning with:
- **Base Model**: Pre-trained on ImageNet (VGG16 or ResNet50)
- **Head**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(3, Softmax)

### Training Strategy

1. **Feature Extraction Phase**:
   - Base model frozen
   - Only head layers trained
   - Learning rate: 1e-4
   - Batch size: 16
   - Early stopping with patience=5

2. **Fine-Tuning Phase**:
   - Last convolutional block unfrozen
   - Lower learning rate: 1e-5
   - Batch size: 16
   - Early stopping with patience=3

### Data Preprocessing

- Convert to grayscale then back to RGB (3 channels)
- Resize to 224×224 pixels
- Normalize to [0, 1]
- Data augmentation (training only):
  - Random horizontal/vertical flips
  - Random brightness/contrast adjustments
  - Random crop

## 📊 Evaluation Metrics

The evaluation script provides:
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: 3×3 matrix showing true vs predicted classes
- **ROC Curve**: Receiver Operating Characteristic curve (malignant vs others)
- **AUC Score**: Area Under the ROC Curve

## 🎨 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the image that the model focuses on when making predictions. This provides interpretability and helps understand model decisions.

- **VGG16 Layer**: `block5_conv3`
- **ResNet50 Layer**: `conv5_block3_out`

## 📈 Results

After training and evaluation, you can expect:
- Model accuracy on test set
- Per-class performance metrics
- Visual confusion matrices
- ROC curves with AUC scores
- Real-time predictions with confidence scores

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure you've trained the models first
   - Check that model files exist in `models/` directory

2. **Dataset path error**:
   - Verify the dataset path in `data_utils.py`
   - Ensure all three folders (normal, benign, malignant) exist

3. **Grad-CAM errors**:
   - Make sure the model is fully trained
   - Check that layer names are correct for your TensorFlow version

4. **Memory errors during training**:
   - Reduce batch size in `get_datasets()` function
   - Use a GPU if available

## 📝 Notes

- The project uses a fixed random seed (42) for reproducibility
- Models are saved in H5 format for compatibility
- All images are preprocessed consistently between training and inference
- The dashboard caches models for faster loading

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Report issues
- Suggest improvements
- Fork and modify for your needs

## 📄 License

This project is for educational and research purposes.

## 👤 Author

Created for breast ultrasound classification research.

---

**For detailed quick start instructions, see [QUICKSTART.md](QUICKSTART.md)**

