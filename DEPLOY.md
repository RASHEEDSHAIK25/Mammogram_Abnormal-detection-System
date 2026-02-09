# Deployment Guide – Mammographic Abnormalities Classifier

Your Streamlit app is **ready to deploy for free** to multiple platforms. Choose one:

---

## 🚀 **Option 1: Streamlit Cloud (Recommended – Easiest)**

**Platform:** Streamlit Cloud (Official, Free tier available)  
**Cost:** Free (with limits: 1 app, 3 GB memory)  
**Setup time:** 2 minutes

### Steps:

1. **Ensure code is on GitHub** (already done ✓)
   ```bash
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select:
     - **Repository:** Your GitHub repo (e.g., `username/Mammogram_Abnormal`)
     - **Branch:** `main`
     - **Main file path:** `app.py`
   - Click "Deploy"

3. **Done!** App is now live at `https://username-mammogram.streamlit.app`

---

## 🐳 **Option 2: Docker + Any VPS (Full Control)**

**Platform:** AWS EC2, DigitalOcean, Linode, etc. (free tier or $5/month)  
**Cost:** $0–$5/month  
**Setup time:** 10 minutes

### Dockerfile:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy:
```bash
# Build image
docker build -t mammogram-app .

# Run locally
docker run -p 8501:8501 mammogram-app

# Or push to Docker Hub and deploy anywhere
docker tag mammogram-app username/mammogram-app
docker push username/mammogram-app
```

---

## 🌐 **Option 3: Render (Free Tier)**

**Platform:** Render (Free tier, auto-deploys from GitHub)  
**Cost:** Free  
**Setup time:** 5 minutes

### Steps:
1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
   - **Port:** 8501
5. Deploy

---

## ⚡ **Option 4: Heroku (Legacy – $7/month minimum now)**

**Cost:** ~$7/month (no more free tier)  
**Setup time:** 5 minutes

Create `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Then:
```bash
heroku login
heroku create mammogram-app
git push heroku main
```

---

## 📋 **Requirements Met:**

✅ No hardcoded paths  
✅ Relative model paths  
✅ Auto-model download on first run  
✅ CPU-only TensorFlow (no CUDA needed)  
✅ All dependencies in `requirements.txt`  
✅ Error handling & fallbacks  
✅ Production config (`.streamlit/config.toml`)  

---

## 🎯 **Recommendation:**

**Use Streamlit Cloud** – it's:
- Free for 1 app
- Instantly deploys from GitHub
- Auto-updates on each push
- No infrastructure to manage
- Perfect for demos/portfolios

**To deploy right now:**
1. Push this repo to GitHub (if not already)
2. Visit https://share.streamlit.io
3. Click "New app" → select your repo
4. Done!

---

**App will be live in ~1–2 minutes at:** `https://your-username-mammogram.streamlit.app`
