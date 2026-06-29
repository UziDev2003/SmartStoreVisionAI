# Smart Store Vision AI 🏪

An intelligent retail analytics system using **YOLOv8** for person detection and **ByteTrack** for multi-object tracking.

---

## 🚀 Quick Start

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py

# Open http://localhost:8501
```

**Features:**
- 📹 Upload any video (MP4, AVI, MOV, MKV)
- 🎯 Real-time person detection + tracking
- 🔢 Unique IDs for each person
- 📍 Zone occupancy monitoring
- 🗺️ Live foot traffic heatmap
- ⚠️ Loitering & intrusion alerts

---

## Features

- **Real-time Person Detection**: YOLOv8 for fast, accurate detection
- **Multi-Object Tracking**: ByteTrack algorithm (~80% MOTA)
- **Zone Analytics**: Define zones with dwell time tracking
- **Heatmap Generation**: Visualize foot traffic patterns
- **Behavior Detection**: Loitering, intrusion, congestion alerts
- **Privacy Compliant**: No facial recognition

## Installation

```bash
pip install -r requirements.txt
# Or minimal:
pip install ultralytics opencv-python numpy streamlit
```

## Usage

### Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```

### CLI
```bash
python main.py --video video.mp4 --show
python main.py --rtsp rtsp://camera:554/stream --show
```

## Performance

| Hardware | Model | FPS |
|----------|-------|-----|
| RTX 3080 | YOLOv8s | ~100 |
| CPU | YOLOv8n | ~15-20 |
| Jetson Orin | YOLOv8n | ~30+ |

## License

MIT