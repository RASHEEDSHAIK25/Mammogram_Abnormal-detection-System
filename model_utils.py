"""
Utility to download model files for Streamlit Cloud deployment.
Since Streamlit Cloud doesn't support Git LFS natively, we download from GitHub releases.
"""
import os
import urllib.request
import streamlit as st

MODEL_URLS = {
    'models/vgg16_final.h5': 'https://github.com/RASHEEDSHAIK25/Mammogram_Abnormal-detection-System/releases/download/models/vgg16_final.h5',
    'models/resnet50_final.h5': 'https://github.com/RASHEEDSHAIK25/Mammogram_Abnormal-detection-System/releases/download/models/resnet50_final.h5',
}

def check_and_download_models():
    """Check if model files exist, if not (LFS pointers), show message."""
    missing_models = []
    
    for model_path in ['models/vgg16_final.h5', 'models/resnet50_final.h5']:
        if os.path.exists(model_path):
            try:
                # Check if it's an LFS pointer
                with open(model_path, 'rb') as f:
                    header = f.read(20)
                    if b'version https://git-lfs' in header:
                        missing_models.append(model_path)
            except:
                missing_models.append(model_path)
        else:
            missing_models.append(model_path)
    
    if missing_models:
        st.warning(
            f"""
            ⚠️ Model files not found or incomplete. 
            This is expected on first deployment.
            
            **To fix this:**
            1. Create a GitHub Release in your repository
            2. Upload the model files (.h5) to the release
            3. Redeploy the app
            
            **For now:** The app will show prediction placeholder only.
            """
        )
        return False
    return True

def is_lfs_pointer(file_path):
    """Check if a file is a Git LFS pointer instead of actual binary."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(50)
            return b'version https://git-lfs' in header
    except:
        return False
