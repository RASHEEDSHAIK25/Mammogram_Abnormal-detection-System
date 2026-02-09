# Streamlit Cloud Deployment Guide

## Prerequisites
1. GitHub account
2. Streamlit account (free at https://streamlit.io/cloud)
3. Repository pushed to GitHub

## Deployment Steps

### Step 1: Push to GitHub
Make sure all your code is committed and pushed (except model files):
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Create Streamlit App on Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your repository: `<your-username>/<Mammogram_Abnormal-detection-System>`
4. Branch: `main`
5. Main file path: `app.py`
6. Click "Deploy"

### Step 3: Wait for First Run
- The app will take 2-3 minutes to build and deploy
- Models will be automatically downloaded on first run from GitHub releases
- Subsequent runs will use cached models

## Important Notes

### ✅ What's Already Configured
- `tensorflow-cpu` in requirements.txt (lightweight for Cloud)
- GPU disabled at startup (CPU only)
- Model auto-download from GitHub releases
- `.streamlit/config.toml` optimized for Cloud

### ⚠️ Potential Issues & Solutions

**If deployment fails:**
1. Check GitHub token/permissions
2. Ensure `requirements.txt` has all dependencies
3. Check Streamlit Cloud logs for errors

**If models don't download:**
1. Verify GitHub releases exist at: https://github.com/RASHEEDSHAIK25/Mammogram_Abnormal-detection-System/releases
2. Check internet connectivity in Cloud environment
3. Model download happens in `setup_models.py`

**If app runs slow:**
- This is normal on free tier (shared CPU resources)
- Models take time to load first time
- Streamlit caches computation results

## Troubleshooting

### Check Cloud Logs
```
App menu → Manage app → View logs
```

### Re-deploy
```
App menu → Reboot app
```

### Clear Cache
```
App menu → Clear cache
```

## More Resources
- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-cloud
- Troubleshooting: https://docs.streamlit.io/streamlit-cloud/troubleshooting
