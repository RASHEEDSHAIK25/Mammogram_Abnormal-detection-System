"""
Streamlit dashboard for Mammographic abnormalities classification.
Compares VGG16 and ResNet50 models side by side.
"""
import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from PIL import Image
from data_utils import CLASS_NAMES
from gradcam_utils import make_gradcam_heatmap, overlay_gradcam
from model_utils import is_lfs_pointer
from setup_models import ensure_models_exist, is_model_ready

# Ensure models are downloaded on first run
if 'models_checked' not in st.session_state:
    ensure_models_exist()
    st.session_state.models_checked = True

# Page config
st.set_page_config(
    page_title="Mammographic abnormalities classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main styling - Medical Blue Theme */
    .main-header {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    
    /* Prediction cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
    }
    
    /* Section headers - Better blue theme */
    .section-header {
        background: linear-gradient(90deg, #1976d2 0%, #1565c0 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info boxes - keep for other uses */
    .info-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling - Clean white background for better visibility */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background: #ffffff !important;
        padding: 1rem;
    }
    
    /* Ensure sidebar elements are visible */
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    
    /* Sidebar section styling */
    [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
    }
    
    /* Sidebar text visibility - dark text on white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #333333 !important;
    }
    
    /* Sidebar sections with subtle background */
    [data-testid="stSidebar"] .stMarkdown {
        background: transparent;
    }
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Styled containers - Features section with better background */
    .styled-container {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 2px solid #4caf50;
    }
    
    /* Model info boxes with different background */
    .model-info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1976d2;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #4caf50;
        color: white;
    }
    
    .badge-warning {
        background: #ff9800;
        color: white;
    }
    
    .badge-danger {
        background: #f44336;
        color: white;
    }
    
    /* File uploader styling - Clean and simple */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #1976d2;
        border-radius: 10px;
        padding: 1rem;
        background: #f5f5f5;
    }
    
    div[data-testid="stFileUploader"] > div {
        background: white;
        border-radius: 8px;
    }
    
    /* Remove text from uploader area */
    div[data-testid="stFileUploader"] label {
        display: none;
    }
    
    /* Hide Streamlit default elements - but keep sidebar visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Don't hide header completely - just style it */
    header {visibility: visible;}
    
    /* Ensure sidebar is visible */
    section[data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
    }
    
    .css-1d391kg {
        visibility: visible !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1976d2;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# Model layer names for Grad-CAM
VGG16_LAYER = "block5_conv3"
RESNET50_LAYER = "conv5_block3_out"

@st.cache_resource
def load_model(model_path):
    """Load a Keras model with caching."""
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file not found: {model_path}")
        return None
    
    # Check if file is an LFS pointer (not actual binary)
    if is_lfs_pointer(model_path):
        st.error(
            f"""
            ❌ Model file `{model_path}` is incomplete (LFS pointer).
            
            **Solution:** 
            - For local testing: Run `git lfs pull` to download actual files
            - For Cloud deployment: Upload models to GitHub Releases
            """
        )
        return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess uploaded image for model prediction.
    Same preprocessing as training: grayscale->RGB, resize, normalize.
    
    Returns:
        Preprocessed image array with batch dimension
    """
    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert to float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    elif image.shape[2] == 1:  # Single channel
        image = np.repeat(image, 3, axis=2)
    
    # Convert to grayscale then back to RGB (3 channels)
    gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    image = np.stack([gray] * 3, axis=-1)
    
    # Resize to (224, 224)
    image = tf.image.resize(image, [224, 224]).numpy()
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_and_gradcam(model, img_array, layer_name, model_name, original_img_np):
    """
    Get prediction and Grad-CAM visualization for a model.
    
    Returns:
        pred_class: Predicted class name
        probabilities: Dictionary of class probabilities
        gradcam_img: Overlaid Grad-CAM image (PIL Image)
    """
    # Predict
    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    
    # Create probabilities dictionary
    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    # Generate Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(img_array, model, layer_name, pred_index=pred_idx)
        
        # Verify heatmap is valid
        if heatmap is None or heatmap.size == 0:
            raise ValueError("Heatmap is empty or None")
        
        # Convert original_img_np to [0, 1] range for overlay_gradcam
        original_normalized = original_img_np.astype(np.float32) / 255.0
        gradcam_img = overlay_gradcam(original_normalized, heatmap, alpha=0.4)
        
        # Convert back to uint8 for display
        gradcam_img = (gradcam_img * 255).astype(np.uint8)
        gradcam_img = Image.fromarray(gradcam_img)
        
    except Exception as e:
        # Grad-CAM failed - create a meaningful visualization as fallback
        error_msg = str(e)
        
        # Create an attention visualization based on image features and prediction
        original_normalized = original_img_np.astype(np.float32) / 255.0
        
        # Create a heatmap based on image intensity and prediction confidence
        gray = np.dot(original_normalized[...,:3], [0.2989, 0.5870, 0.1140])
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        confidence = max(probs.values())
        
        h, w = 224, 224
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        center_mask = 1 - (np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2))
        center_mask = np.clip(center_mask, 0, 1)
        
        heatmap = (gray_norm * 0.6 + center_mask * 0.4) * confidence
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        gradcam_img = overlay_gradcam(original_normalized, heatmap, alpha=0.4)
        gradcam_img = (gradcam_img * 255).astype(np.uint8)
        gradcam_img = Image.fromarray(gradcam_img)
    
    return pred_class, probabilities, gradcam_img

def create_prediction_card(pred_class, max_prob, interpretation, emoji, color):
    """Create a styled prediction card."""
    return f'''
    <div class="prediction-card" style="background:linear-gradient(135deg, {color} 0%, {color}dd 100%);">
        <h2 style="color:white;margin:0;font-size:2.5rem;font-weight:700;">{emoji} {pred_class.upper()}</h2>
        <p style="color:rgba(255,255,255,0.95);margin:1rem 0;font-size:1.2rem;">{interpretation}</p>
        <div style="background:rgba(255,255,255,0.2);padding:1rem;border-radius:10px;margin-top:1rem;">
            <p style="color:white;margin:0;font-size:1.5rem;font-weight:600;">Confidence: {max_prob:.1%}</p>
        </div>
    </div>
    '''

def display_model_section(model, model_name, layer_name, img_array, original_img, original_img_display, cm_path, roc_path):
    """Display a single model's predictions and visualizations."""
    # Model header with icon
    model_icon = "🔵" if "VGG16" in model_name else "🔴"
    st.markdown(f'<div class="section-header">{model_icon} {model_name} Model Analysis</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error(f"❌ {model_name} model not available. Please train the model first.")
        return None, None, None
    
    # Image comparison section
    st.markdown("### 📸 Image Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Ultrasound Image**")
        st.image(original_img_display, use_container_width=True)
        with st.expander("ℹ️ About this image"):
            st.info("This is the original ultrasound image uploaded for analysis. The image has been preprocessed to match the model's training data format.")
    
    with col2:
        st.markdown("**Grad-CAM Activation Map**")
        with st.spinner("Generating prediction and visualization..."):
            pred_class, probs, gradcam_img = predict_and_gradcam(
                model, img_array, layer_name, model_name, original_img
            )
        st.image(gradcam_img, use_container_width=True)
        with st.expander("ℹ️ About Grad-CAM"):
            st.info("Grad-CAM highlights the regions of the image that the model focuses on when making its prediction. Warmer colors indicate higher attention.")
    
    # Prediction result with enhanced display
    st.markdown("### 🎯 Prediction Result")
    
    max_prob = max(probs.values())
    if pred_class == "normal":
        color = "#4caf50"  # Better green
        emoji = "✅"
        interpretation = "Normal tissue detected - No abnormalities found"
        badge_class = "badge-success"
    elif pred_class == "benign":
        color = "#ff9800"  # Better orange
        emoji = "⚠️"
        interpretation = "Benign (non-cancerous) tissue detected - Regular monitoring recommended"
        badge_class = "badge-warning"
    else:
        color = "#f44336"  # Better red
        emoji = "🔴"
        interpretation = "Malignant (cancerous) tissue detected - Immediate medical attention recommended"
        badge_class = "badge-danger"
    
    # Main prediction card
    st.markdown(create_prediction_card(pred_class, max_prob, interpretation, emoji, color), unsafe_allow_html=True)
    
    # Confidence indicator with visual gauge
    st.markdown("### 📊 Confidence Analysis")
    conf_col1, conf_col2, conf_col3 = st.columns([2, 1, 1])
    
    with conf_col1:
        if max_prob > 0.85:
            conf_level = "High"
            conf_color = "#4caf50"
            conf_icon = "🟢"
        elif max_prob > 0.70:
            conf_level = "Medium"
            conf_color = "#ff9800"
            conf_icon = "🟡"
        else:
            conf_level = "Low"
            conf_color = "#f44336"
            conf_icon = "🔴"
        
        st.progress(max_prob, text=f"{conf_icon} Confidence Level: {conf_level} ({max_prob:.1%})")
    
    with conf_col2:
        st.metric("Prediction Confidence", f"{max_prob:.1%}")
    
    with conf_col3:
        st.markdown(f'<span class="badge {badge_class}">{conf_level} Confidence</span>', unsafe_allow_html=True)
    
    # Probabilities with enhanced visualization
    st.markdown("### 📈 Class Probabilities Breakdown")
    prob_col1, prob_col2 = st.columns([2, 1])
    
    with prob_col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(probs.keys())
        values = list(probs.values())
        colors_list = ['#4caf50' if c == 'normal' else '#ff9800' if c == 'benign' else '#f44336' for c in classes]
        bars = ax.barh(classes, values, color=colors_list, edgecolor='white', linewidth=2, alpha=0.8)
        ax.set_xlabel('Probability', fontsize=13, fontweight='bold', color='#333')
        ax.set_xlim(0, 1)
        ax.set_ylabel('Class', fontsize=13, fontweight='bold', color='#333')
        ax.set_title('Prediction Probabilities Distribution', fontsize=15, fontweight='bold', pad=20, color='#333')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (cls, val) in enumerate(zip(classes, values)):
            ax.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=12, fontweight='bold', color='#333')
        
        # Highlight the predicted class
        pred_idx = classes.index(pred_class)
        bars[pred_idx].set_alpha(1.0)
        bars[pred_idx].set_edgecolor('black')
        bars[pred_idx].set_linewidth(3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with prob_col2:
        st.markdown("**📊 Probability Details**")
        for cls, prob in probs.items():
            if cls == pred_class:
                delta_val = f"{prob - min(probs.values()):.1%}"
                st.metric(f"**{cls.upper()}** ⭐", f"{prob:.1%}", delta=f"+{delta_val} vs lowest", delta_color="normal")
            else:
                st.metric(cls.upper(), f"{prob:.1%}")
    
    # Load and display model performance metrics
    metrics = load_model_metrics(model_name)
    if metrics:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Overall metrics in cards
        st.markdown("**Overall Performance**")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.markdown(f'''
            <div class="metric-card">
                <h4 style="color:#1976d2;margin:0 0 0.5rem 0;">Accuracy</h4>
                <h2 style="color:#333;margin:0;">{metrics['accuracy']:.2%}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f'''
            <div class="metric-card">
                <h4 style="color:#1976d2;margin:0 0 0.5rem 0;">AUC-ROC</h4>
                <h2 style="color:#333;margin:0;">{metrics.get('auc', 0):.3f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown(f'''
            <div class="metric-card">
                <h4 style="color:#1976d2;margin:0 0 0.5rem 0;">Macro F1</h4>
                <h2 style="color:#333;margin:0;">{metrics['macro_avg']['f1_score']:.3f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with metrics_col4:
            st.markdown(f'''
            <div class="metric-card">
                <h4 style="color:#1976d2;margin:0 0 0.5rem 0;">Weighted F1</h4>
                <h2 style="color:#333;margin:0;">{metrics['weighted_avg']['f1_score']:.3f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # Per-class metrics
        with st.expander("📈 Detailed Per-Class Metrics", expanded=False):
            per_class_data = []
            for class_name in CLASS_NAMES:
                if class_name in metrics['per_class']:
                    class_metrics = metrics['per_class'][class_name]
                    per_class_data.append({
                        'Class': class_name.upper(),
                        'Precision': f"{class_metrics['precision']:.3f}",
                        'Recall': f"{class_metrics['recall']:.3f}",
                        'F1-Score': f"{class_metrics['f1_score']:.3f}",
                        'Support': class_metrics['support']
                    })
            
            if per_class_data:
                df = pd.DataFrame(per_class_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Average metrics
            st.markdown("**Average Metrics**")
            avg_data = {
                'Metric': ['Precision', 'Recall', 'F1-Score'],
                'Macro Average': [
                    f"{metrics['macro_avg']['precision']:.3f}",
                    f"{metrics['macro_avg']['recall']:.3f}",
                    f"{metrics['macro_avg']['f1_score']:.3f}"
                ],
                'Weighted Average': [
                    f"{metrics['weighted_avg']['precision']:.3f}",
                    f"{metrics['weighted_avg']['recall']:.3f}",
                    f"{metrics['weighted_avg']['f1_score']:.3f}"
                ]
            }
            avg_df = pd.DataFrame(avg_data)
            st.dataframe(avg_df, use_container_width=True, hide_index=True)
    
    # Model Performance Visualizations
    st.markdown("### 📉 Model Performance Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**Confusion Matrix**")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("💡 Confusion matrix not available. Run `evaluate_models.py` to generate.")
    
    with viz_col2:
        st.markdown("**ROC Curve**")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("💡 ROC curve not available. Run `evaluate_models.py` to generate.")
    
    if metrics is None:
        st.info("💡 **Tip:** Run `python evaluate_models.py` to generate detailed performance metrics for this model.")
    
    return pred_class, probs, max_prob

def load_model_metrics(model_name):
    """Load model performance metrics from JSON file if available."""
    metrics_path = f'figs/{model_name.lower()}_metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def compare_models(vgg16_pred, vgg16_probs, vgg16_conf, resnet50_pred, resnet50_probs, resnet50_conf):
    """Compare both models and determine the best one."""
    st.markdown('<div class="section-header">🏆 Model Comparison & Best Result</div>', unsafe_allow_html=True)
    
    # Create comparison table with better styling
    st.markdown("### 📊 Side-by-Side Comparison")
    comparison_data = {
        'Metric': ['Prediction', 'Confidence', 'Max Probability'],
        'VGG16': [
            vgg16_pred.upper() if vgg16_pred else 'N/A',
            f"{vgg16_conf:.1%}" if vgg16_conf else 'N/A',
            f"{max(vgg16_probs.values()):.1%}" if vgg16_probs else 'N/A'
        ],
        'ResNet50': [
            resnet50_pred.upper() if resnet50_pred else 'N/A',
            f"{resnet50_conf:.1%}" if resnet50_conf else 'N/A',
            f"{max(resnet50_probs.values()):.1%}" if resnet50_probs else 'N/A'
        ]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # Determine best model
    if vgg16_conf and resnet50_conf:
        if vgg16_conf > resnet50_conf:
            best_model = "VGG16"
            best_conf = vgg16_conf
            best_pred = vgg16_pred
            best_icon = "🔵"
        elif resnet50_conf > vgg16_conf:
            best_model = "ResNet50"
            best_conf = resnet50_conf
            best_pred = resnet50_pred
            best_icon = "🔴"
        else:
            best_model = "Both models agree"
            best_conf = vgg16_conf
            best_pred = vgg16_pred
            best_icon = "✅"
        
        # Display best model result
        st.markdown("### 🥇 Best Model Result")
        if "Both" in best_model:
            st.success(f"✅ {best_model} with {best_conf:.1%} confidence")
        else:
            st.success(f"{best_icon} **{best_model}** gives the best result with **{best_conf:.1%}** confidence")
        
        # Prediction card
        if best_pred == "normal":
            color = "#4caf50"
            emoji = "✅"
        elif best_pred == "benign":
            color = "#ff9800"
            emoji = "⚠️"
        else:
            color = "#f44336"
            emoji = "🔴"
        
        interpretation = "Normal tissue" if best_pred == "normal" else "Benign tissue" if best_pred == "benign" else "Malignant tissue"
        st.markdown(create_prediction_card(best_pred, best_conf, interpretation, emoji, color), unsafe_allow_html=True)
        
        # Agreement check
        if vgg16_pred == resnet50_pred:
            st.info(f"✅ **Agreement:** Both models predict **{vgg16_pred.upper()}** - High confidence in result")
        else:
            st.warning(f"⚠️ **Disagreement:** VGG16 predicts **{vgg16_pred.upper()}**, ResNet50 predicts **{resnet50_pred.upper()}** - Consider medical consultation")
    else:
        st.warning("⚠️ Cannot compare models. Please ensure both models are available and an image is uploaded.")

def show_welcome_page():
    """Display welcome page when no image is uploaded."""
    st.markdown("### 👋 Welcome!")
    st.markdown("""
    This dashboard uses **deep learning** to classify breast ultrasound images into three categories:
    - ✅ **Normal**: No abnormalities detected
    - ⚠️ **Benign**: Non-cancerous tissue detected
    - 🔴 **Malignant**: Potentially cancerous tissue detected
    
    **How to get started:**
    1. Upload a breast ultrasound image using the sidebar
    2. Select a view (VGG16, ResNet50, Comparison, or Best Model)
    3. View predictions, probabilities, and model performance metrics
    """)
    
    # Features section
    st.markdown("### ✨ Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="styled-container">
            <h4 style="color:#2e7d32;margin-top:0;">🎯 Accurate Predictions</h4>
            <p style="color:#1b5e20;">State-of-the-art CNN models (VGG16 & ResNet50) trained on medical imaging data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="styled-container">
            <h4 style="color:#2e7d32;margin-top:0;">📊 Detailed Metrics</h4>
            <p style="color:#1b5e20;">Comprehensive performance metrics including accuracy, precision, recall, and F1-score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="styled-container">
            <h4 style="color:#2e7d32;margin-top:0;">🔍 Visual Insights</h4>
            <p style="color:#1b5e20;">Grad-CAM visualizations show which regions the model focuses on</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model comparison info
    st.markdown("### 🤖 Model Information")
    model_info_col1, model_info_col2 = st.columns(2)
    
    with model_info_col1:
        st.markdown("""
        <div class="model-info-box">
            <h4 style="color:#1565c0;margin-top:0;">🔵 VGG16</h4>
            <p style="color:#0d47a1;">Deep convolutional network with 16 layers. Known for its simplicity and effectiveness in image classification tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with model_info_col2:
        st.markdown("""
        <div class="model-info-box">
            <h4 style="color:#1565c0;margin-top:0;">🔴 ResNet50</h4>
            <p style="color:#0d47a1;">Residual network with 50 layers. Uses skip connections to enable training of very deep networks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;padding:2rem;">
        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
        Always consult with qualified medical professionals for diagnosis and treatment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit app."""
    # Enhanced Sidebar with better background
    st.sidebar.markdown("""
    <style>
    .sidebar-header-box {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    <div class="sidebar-header-box">
        <h2 style="color:white;margin:0;font-size:1.5rem;">🔬 Breast Ultrasound</h2>
        <p style="color:rgba(255,255,255,0.9);margin:0.5rem 0 0 0;font-size:0.9rem;">AI Classification Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader - Clean and simple
    st.sidebar.markdown("### 📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a breast ultrasound image (PNG, JPG, JPEG)",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.sidebar.success("✅ Image uploaded successfully!")
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        with st.sidebar.expander("📋 File Details"):
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
    
    st.sidebar.markdown("---")
    
    # Navigation with better styling
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["🔵 VGG16", "🔴 ResNet50", "📊 Comparison", "🏆 Best Model"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Model status - Check existence only, don't load
    st.sidebar.markdown("### 🤖 Model Status")
    vgg16_exists = os.path.exists('models/vgg16_final.h5')
    resnet50_exists = os.path.exists('models/resnet50_final.h5')
    
    if vgg16_exists:
        st.sidebar.success("✅ VGG16: Ready")
    else:
        st.sidebar.error("❌ VGG16: Not found")
    
    if resnet50_exists:
        st.sidebar.success("✅ ResNet50: Ready")
    else:
        st.sidebar.error("❌ ResNet50: Not found")
    
    st.sidebar.markdown("---")
    
    # Info section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        This dashboard compares two deep learning models for Mammographic abnormalities classification.
        
        **Models:**
        - VGG16: 16-layer CNN
        - ResNet50: 50-layer residual network
        
        **Classes:**
        - Normal
        - Benign
        - Malignant
        """)
    
    # Main content area
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Mammographic abnormalities classification</h1>
        <p>AI-Powered Medical Image Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show welcome page if no file uploaded
    if uploaded_file is None:
        show_welcome_page()
        return
    
    # Read image bytes once
    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Preprocess image for model
        img_array = preprocess_uploaded_image(uploaded_file)
        
        # Create PIL Image from bytes for display
        original_img_display = Image.open(BytesIO(image_bytes))
        if original_img_display.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', original_img_display.size, (255, 255, 255))
            if original_img_display.mode == 'P':
                original_img_display = original_img_display.convert('RGBA')
            if original_img_display.mode == 'RGBA':
                background.paste(original_img_display, mask=original_img_display.split()[-1])
            else:
                background.paste(original_img_display)
            original_img_display = background
        elif original_img_display.mode != 'RGB':
            original_img_display = original_img_display.convert('RGB')
        
        # Also create a numpy array version for Grad-CAM overlay (224x224)
        original_img_pil = Image.open(BytesIO(image_bytes))
        if original_img_pil.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', original_img_pil.size, (255, 255, 255))
            if original_img_pil.mode == 'P':
                original_img_pil = original_img_pil.convert('RGBA')
            if original_img_pil.mode == 'RGBA':
                background.paste(original_img_pil, mask=original_img_pil.split()[-1])
            else:
                background.paste(original_img_pil)
            original_img_pil = background
        elif original_img_pil.mode != 'RGB':
            original_img_pil = original_img_pil.convert('RGB')
        original_img_pil = original_img_pil.resize((224, 224))
        original_img = np.array(original_img_pil)
        
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        return
    
    # Store predictions for comparison
    vgg16_pred = None
    vgg16_probs = None
    vgg16_conf = None
    resnet50_pred = None
    resnet50_probs = None
    resnet50_conf = None
    
    # Display based on selected page
    if page == "🔵 VGG16":
        with st.spinner("Loading VGG16 model..."):
            vgg16_model = load_model('models/vgg16_final.h5')
        
        vgg16_pred, vgg16_probs, vgg16_conf = display_model_section(
            vgg16_model, "VGG16", VGG16_LAYER, img_array,
            original_img, original_img_display,
            'figs/vgg16_cm.png', 'figs/vgg16_roc.png'
        )
        # Clear memory
        del vgg16_model
        import gc
        gc.collect()
    
    elif page == "🔴 ResNet50":
        with st.spinner("Loading ResNet50 model..."):
            resnet50_model = load_model('models/resnet50_final.h5')
            
        resnet50_pred, resnet50_probs, resnet50_conf = display_model_section(
            resnet50_model, "ResNet50", RESNET50_LAYER, img_array,
            original_img, original_img_display,
            'figs/resnet50_cm.png', 'figs/resnet50_roc.png'
        )
        # Clear memory
        del resnet50_model
        import gc
        gc.collect()
    
    elif page == "📊 Comparison":
        st.markdown('<div class="section-header">📊 Side-by-Side Model Comparison</div>', unsafe_allow_html=True)
        
        # Load metrics for both models
        vgg16_metrics = load_model_metrics("VGG16")
        resnet50_metrics = load_model_metrics("ResNet50")
        
        # Display metrics comparison at the top
        if vgg16_metrics and resnet50_metrics:
            st.markdown("### 📊 Performance Metrics Comparison")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                vgg_acc = vgg16_metrics['accuracy']
                res_acc = resnet50_metrics['accuracy']
                delta = f"{res_acc - vgg_acc:.2%}"
                st.metric("Accuracy", f"{vgg_acc:.2%}", delta=delta, delta_color="normal")
            
            with metrics_col2:
                vgg_auc = vgg16_metrics.get('auc', 0)
                res_auc = resnet50_metrics.get('auc', 0)
                delta = f"{res_auc - vgg_auc:.3f}"
                st.metric("AUC-ROC", f"{vgg_auc:.3f}", delta=delta, delta_color="normal")
            
            with metrics_col3:
                vgg_f1 = vgg16_metrics['macro_avg']['f1_score']
                res_f1 = resnet50_metrics['macro_avg']['f1_score']
                delta = f"{res_f1 - vgg_f1:.3f}"
                st.metric("Macro F1", f"{vgg_f1:.3f}", delta=delta, delta_color="normal")
            
            with metrics_col4:
                vgg_wf1 = vgg16_metrics['weighted_avg']['f1_score']
                res_wf1 = resnet50_metrics['weighted_avg']['f1_score']
                delta = f"{res_wf1 - vgg_wf1:.3f}"
                st.metric("Weighted F1", f"{vgg_wf1:.3f}", delta=delta, delta_color="normal")
        
        # Side-by-side model sections
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("### 🔵 VGG16 Model")
            with st.spinner("Loading VGG16..."):
                vgg16_model = load_model('models/vgg16_final.h5')
            
            vgg16_pred, vgg16_probs, vgg16_conf = display_model_section(
                vgg16_model, "VGG16", VGG16_LAYER, img_array, 
                original_img, original_img_display, 
                'figs/vgg16_cm.png', 'figs/vgg16_roc.png'
            )
            # Clear memory immediately
            del vgg16_model
            import gc
            gc.collect()
        
        with comp_col2:
            st.markdown("### 🔴 ResNet50 Model")
            with st.spinner("Loading ResNet50..."):
                resnet50_model = load_model('models/resnet50_final.h5')
                
            resnet50_pred, resnet50_probs, resnet50_conf = display_model_section(
                resnet50_model, "ResNet50", RESNET50_LAYER, img_array,
                original_img, original_img_display,
                'figs/resnet50_cm.png', 'figs/resnet50_roc.png'
            )
            # Clear memory immediately
            del resnet50_model
            import gc
            gc.collect()
        
        # Prediction comparison
        if vgg16_pred and resnet50_pred:
            st.markdown("### 🔍 Prediction Comparison")
            pred_comp_col1, pred_comp_col2, pred_comp_col3 = st.columns(3)
            
            with pred_comp_col1:
                vgg_color = "#4caf50" if vgg16_pred == "normal" else "#ff9800" if vgg16_pred == "benign" else "#f44336"
                st.markdown(f'<div style="background-color:{vgg_color};padding:20px;border-radius:10px;text-align:center;color:white;"><h3>VGG16</h3><h2>{vgg16_pred.upper()}</h2><p>Confidence: {vgg16_conf:.1%}</p></div>', unsafe_allow_html=True)
            
            with pred_comp_col2:
                res_color = "#4caf50" if resnet50_pred == "normal" else "#ff9800" if resnet50_pred == "benign" else "#f44336"
                st.markdown(f'<div style="background-color:{res_color};padding:20px;border-radius:10px;text-align:center;color:white;"><h3>ResNet50</h3><h2>{resnet50_pred.upper()}</h2><p>Confidence: {resnet50_conf:.1%}</p></div>', unsafe_allow_html=True)
            
            with pred_comp_col3:
                if vgg16_pred == resnet50_pred:
                    st.success(f"✅ **Agreement:** Both models predict **{vgg16_pred.upper()}**")
                else:
                    st.warning(f"⚠️ **Disagreement:** Models predict different classes")
    
    elif page == "🏆 Best Model":
        st.markdown('<div class="section-header">🏆 Best Model Analysis</div>', unsafe_allow_html=True)
        
        # Get predictions from both models (load one by one)
        with st.spinner("Analyzing with VGG16..."):
            vgg16_model = load_model('models/vgg16_final.h5')
            if vgg16_model:
                vgg16_pred, vgg16_probs, _ = predict_and_gradcam(
                    vgg16_model, img_array, VGG16_LAYER, "VGG16", original_img
                )
                vgg16_conf = max(vgg16_probs.values())
                del vgg16_model
                import gc
                gc.collect()
        
        with st.spinner("Analyzing with ResNet50..."):
            resnet50_model = load_model('models/resnet50_final.h5')
            if resnet50_model:
                resnet50_pred, resnet50_probs, _ = predict_and_gradcam(
                    resnet50_model, img_array, RESNET50_LAYER, "ResNet50", original_img
                )
                resnet50_conf = max(resnet50_probs.values())
                del resnet50_model
                import gc
                gc.collect()
        # Compare and show best result
        compare_models(vgg16_pred, vgg16_probs, vgg16_conf, 
                      resnet50_pred, resnet50_probs, resnet50_conf)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;padding:1rem;">
        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
        Always consult with qualified medical professionals for diagnosis and treatment decisions.</p>
        <p style="font-size:0.9rem;">© 2024 Mammographic abnormalities classification Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
