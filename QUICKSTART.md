# Quick Start Guide

This guide will help you get started with the Breast Ultrasound Classification project in just a few steps.

## 🎯 Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **Dataset** located at: `D:\Karishma(ECE)\Dataset_BUSI_with_GT\`
3. **Internet connection** (for downloading pre-trained weights)

## ⚡ 5-Minute Quick Start

### Step 1: Install Dependencies

Open your terminal/command prompt and run:

```bash
pip install tensorflow streamlit pillow numpy scikit-learn matplotlib seaborn opencv-python
```

**Note**: If you have a GPU, install `tensorflow-gpu` instead of `tensorflow` for faster training.

### Step 2: Verify Dataset Structure

Ensure your dataset folder structure looks like this:

```
D:\Karishma(ECE)\Dataset_BUSI_with_GT\
├── normal\
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── benign\
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── malignant\
    ├── image1.png
    ├── image2.png
    └── ...
```

### Step 3: Create Required Directories

```bash
mkdir models figs
```

Or in PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path models,figs
```

### Step 4: Train the Models

#### Option A: Train Both Models (Recommended)

**Train VGG16:**
```bash
python train_vgg16.py
```

This will take some time (30-60 minutes depending on your hardware). You'll see:
- Dataset loading progress
- Training progress for Phase 1 (feature extraction)
- Training progress for Phase 2 (fine-tuning)
- Final validation accuracy

**Train ResNet50:**
```bash
python train_resnet50.py
```

Similar process for ResNet50.

#### Option B: Train One Model First (Faster Start)

If you want to test quickly, train just VGG16 first:
```bash
python train_vgg16.py
```

You can train ResNet50 later.

### Step 5: Evaluate Models

After training, generate evaluation metrics:

```bash
python evaluate_models.py
```

This creates:
- Confusion matrices in `figs/`
- ROC curves in `figs/`
- Prints classification reports to console

### Step 6: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 🎨 Using the Dashboard

### Upload an Image

1. In the sidebar, click **"Upload Ultrasound Image"**
2. Select an image from your dataset or any test image
3. The image will be processed automatically

### Navigate Views

Use the radio buttons in the sidebar to switch between:

1. **📊 Comparison**: See both models side-by-side
2. **🔵 VGG16**: View only VGG16 predictions
3. **🔴 ResNet50**: View only ResNet50 predictions  
4. **🏆 Best Model**: See which model performs better

### What You'll See

For each model:
- **Original Image**: Your uploaded ultrasound image
- **Grad-CAM Heatmap**: Visual explanation showing where the model focuses
- **Prediction**: Normal, Benign, or Malignant
- **Class Probabilities**: Bar chart showing confidence for each class
- **Confidence Level**: High/Medium/Low
- **Confusion Matrix**: Model's overall performance
- **ROC Curve**: Model's discrimination ability

### Best Model Section

The "🏆 Best Model" view shows:
- Comparison table with both models' predictions
- Which model has higher confidence
- Final recommended prediction
- Agreement status (whether models agree or disagree)

## 📋 Complete Workflow Example

Here's a complete example workflow:

```bash
# 1. Install dependencies
pip install tensorflow streamlit pillow numpy scikit-learn matplotlib seaborn opencv-python

# 2. Create directories
mkdir models figs

# 3. Train VGG16 (takes 30-60 minutes)
python train_vgg16.py

# 4. Train ResNet50 (takes 30-60 minutes)
python train_resnet50.py

# 5. Evaluate models
python evaluate_models.py

# 6. Launch dashboard
streamlit run app.py
```

## ⏱️ Time Estimates

- **Dependencies installation**: 2-5 minutes
- **VGG16 training**: 30-60 minutes (CPU) / 10-20 minutes (GPU)
- **ResNet50 training**: 30-60 minutes (CPU) / 10-20 minutes (GPU)
- **Evaluation**: 2-5 minutes
- **Dashboard**: Instant (after models are trained)

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Install missing package
```bash
pip install <package-name>
```

### Issue: "Dataset folder not found"

**Solution**: 
1. Check the path in `data_utils.py` (lines 10-12)
2. Update the path to match your system
3. Ensure all three folders (normal, benign, malignant) exist

### Issue: "Model file not found"

**Solution**: 
1. Make sure you've completed training
2. Check that `models/vgg16_final.h5` and/or `models/resnet50_final.h5` exist
3. Re-run the training script if needed

### Issue: "Out of memory" during training

**Solution**:
1. Reduce batch size in `data_utils.py` (change `batch_size=16` to `batch_size=8`)
2. Close other applications
3. Use a machine with more RAM or use a GPU

### Issue: Streamlit dashboard won't start

**Solution**:
1. Check if port 8501 is already in use
2. Try: `streamlit run app.py --server.port 8502`
3. Make sure Streamlit is installed: `pip install streamlit`

## 💡 Tips for Best Results

1. **Use GPU**: Training is much faster with a GPU
2. **Train both models**: For best comparison results
3. **Run evaluation**: Generate confusion matrices and ROC curves
4. **Test with various images**: Try different images from your dataset
5. **Check Best Model view**: This gives you the most reliable prediction

## 🎓 Understanding the Output

### Prediction Colors
- 🟢 **Green**: Normal (healthy tissue)
- 🟠 **Orange**: Benign (non-cancerous)
- 🔴 **Red**: Malignant (cancerous)

### Confidence Levels
- **High (>85%)**: Very confident prediction
- **Medium (70-85%)**: Moderately confident
- **Low (<70%)**: Less confident, may need review

### Grad-CAM Heatmap
- **Red/Orange areas**: Regions the model focuses on
- **Blue areas**: Less important regions
- Helps understand model reasoning

## 📚 Next Steps

After completing the quick start:

1. **Experiment with different images**: Test the models with various ultrasound images
2. **Review evaluation metrics**: Check confusion matrices and ROC curves
3. **Compare models**: Use the Best Model view to see which performs better
4. **Fine-tune if needed**: Adjust hyperparameters in training scripts
5. **Explore code**: Understand how each component works

## 🆘 Need Help?

If you encounter issues:

1. Check the error message carefully
2. Verify all prerequisites are met
3. Ensure dataset path is correct
4. Check that all dependencies are installed
5. Review the main [README.md](README.md) for detailed documentation

---

**Happy Classifying! 🎉**

For detailed documentation, see [README.md](README.md)

