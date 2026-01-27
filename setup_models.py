"""
Download model files from GitHub releases for Streamlit Cloud deployment.
This runs on first launch to ensure models are available.
"""
import os
import urllib.request
import streamlit as st

# GitHub repo details
GITHUB_REPO = "RASHEEDSHAIK25/Mammogram_Abnormal-detection-System"
RELEASE_TAG = "models"

# Model files to download
MODELS = {
    'models/vgg16_final.h5': f'https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/vgg16_final.h5',
    'models/resnet50_final.h5': f'https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/resnet50_final.h5',
}

def download_model(model_path, url):
    """Download a model file from GitHub release."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        with st.spinner(f"⏳ Downloading {os.path.basename(model_path)}..."):
            urllib.request.urlretrieve(url, model_path)
        st.success(f"✅ Downloaded {os.path.basename(model_path)}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download {model_path}: {str(e)}")
        return False

def ensure_models_exist():
    """Check if models exist, if not (or are LFS pointers), download them."""
    missing_models = {}
    
    for model_path, url in MODELS.items():
        needs_download = False
        
        if not os.path.exists(model_path):
            needs_download = True
        else:
            # Check if it's an LFS pointer (text file) instead of binary
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(50)
                    if b'version https://git-lfs' in header:
                        needs_download = True
            except:
                needs_download = True
        
        if needs_download:
            missing_models[model_path] = url
    
    if missing_models:
        st.warning("📥 First-time setup: Downloading model files...")
        
        for model_path, url in missing_models.items():
            download_model(model_path, url)
        
        st.success("✅ All models downloaded! Refreshing app...")
        st.rerun()

def is_model_ready():
    """Check if all required models are available and ready."""
    for model_path in MODELS.keys():
        if not os.path.exists(model_path):
            return False
        
        # Check for LFS pointers
        try:
            with open(model_path, 'rb') as f:
                header = f.read(50)
                if b'version https://git-lfs' in header:
                    return False
        except:
            return False
    
    return True
