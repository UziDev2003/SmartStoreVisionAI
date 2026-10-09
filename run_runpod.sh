#!/bin/bash
set -e

echo "=================================================="
echo "   SmartStoreVisionAI - RunPod Automated Setup   "
echo "=================================================="

# 1. Update system and install required OpenCV/video libraries
echo "[1/4] Installing system dependencies (ffmpeg, opencv libs, tmux)..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 git tmux psmisc > /dev/null 2>&1
# Remove Debian-managed blinker to prevent pip uninstall error
apt-get remove -y -qq python3-blinker > /dev/null 2>&1 || true

# 2. Upgrade pip and install all Python requirements
echo "[2/4] Installing all Python dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install --ignore-installed blinker
python3 -m pip install -r requirements.txt

# 3. Configure Streamlit for RunPod proxy (disable CORS & XSRF)
echo "[3/4] Configuring Streamlit for RunPod proxy..."
mkdir -p ~/.streamlit
cat << 'EOF' > ~/.streamlit/config.toml
[server]
port = 8888
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

# 4. Stop any process occupying port 8888 (e.g. default Jupyter)
echo "[4/4] Freeing port 8888..."
pkill -f jupyter || true
sleep 1

echo "=================================================="
echo "🚀 Starting Smart Store Vision AI on port 8888..."
echo "👉 View in browser: RunPod Console -> Connect -> Port 8888"
echo "=================================================="

python3 -m streamlit run streamlit_app.py
