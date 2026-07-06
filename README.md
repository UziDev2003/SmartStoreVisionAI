# Smart Store Vision AI 🏪

An intelligent retail analytics and **suspicious-activity detection** system built on **YOLOv8** + **ByteTrack**, with an interactive Streamlit UI for drawing zones directly on video, live alerting, and ML-based false-positive suppression.

---

## ✨ Features

### 🎯 Core Pipeline
- **YOLOv8** person detection (with optional bag/backpack/handbag classes)
- **ByteTrack** multi-object tracking
- **Zone-based analytics** with dwell time, occupancy, congestion
- **Heatmap generation** with hotspot detection
- **Privacy compliant**: no facial recognition

### 🚨 Suspicious Activity Detection (Task 6)
All nine required detectors, with **low-false-positive** guards built in:

| # | Detector | Description |
|---|----------|-------------|
| 1 | **Loitering Detection** | Long-dwell + low motion-variance (suppresses browsing FPs) |
| 2 | **Restricted Zone Violation** | Entry into any `restricted` zone; severity escalation on prior approach |
| 3 | **Shelf Tampering Detection** | Combines motion intensity + MediaPipe hand-near-shelf + (optional) bag class |
| 4 | **Abnormal Shelf Interaction** | Excessive dwell or repeated revisits to a `shelf` zone |
| 5 | **Unattended Checkout Detection** | Stationary object/pixel cluster in a `checkout` zone after person leaves; uses YOLO bag-class when available |
| 6 | **Suspicious Movement Analysis** | Running (peak speed) and erratic motion (heading change count) |
| 7 | **Rule-Based Anomaly Detection** | YAML-driven composable rules evaluated per-track |
| 8 | **Configurable Alert Thresholds** | Every detector and the global guard is tunable from `config/suspicious_activity.yaml` and live-tunable in the UI |
| 9 | **Low False-Positive Detection** | Multi-frame confirmation, hysteresis, per-rule cooldowns, evidence-signature dedup, scikit-learn IsolationForest gating |

### 🖥️ Interactive UI (Streamlit)
- **📹 Live** — process uploaded videos with full pipeline
- **🗺️ Zone Editor** — *draw polygons directly on the video frame* using `streamlit-drawable-canvas`
- **🚨 Suspicious Activity** — live alert feed with filters, severity badges, evidence drill-down
- **🎛️ Threshold Tuner** — live sliders that write override YAML
- **📊 Dashboard** — quick stats and metrics

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│              Main Pipeline (main.py)                       │
└─────────────────────────┬──────────────────────────────────┘
                          │ tracks + zones + frames
                          ▼
┌────────────────────────────────────────────────────────────┐
│   NEW: src/analytics/suspicious_activity.py                │
│   ┌────────────────────────────────────────────────┐       │
│   │ SuspiciousActivityDetector (orchestrator)      │       │
│   │  ├─ LoiteringDetector   (motion-variance)      │       │
│   │  ├─ RestrictedZoneDetector (severity escal.)  │       │
│   │  ├─ ShelfTamperingDetector (motion + pose)     │       │
│   │  ├─ ShelfInteractionDetector                   │       │
│   │  ├─ UnattendedCheckoutDetector (bag-class)     │       │
│   │  ├─ MovementAnalyzer (running/erratic)         │       │
│   │  ├─ PoseAnalyzer (MediaPipe)                   │       │
│   │  ├─ MultiCameraCorrelator (Re-ID)              │       │
│   │  ├─ AnomalyScorer (IsolationForest gating)     │       │
│   │  └─ RuleEngine (YAML rules)                    │       │
│   └────────────────────────────────────────────────┘       │
└─────────────────────────┬──────────────────────────────────┘
                          │ SuspiciousActivity (typed)
                          ▼
┌────────────────────────────────────────────────────────────┐
│   AlertManager (extended) → snapshots + callbacks          │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Industry-Standard Libraries

| Library | Purpose | Why |
|---|---|---|
| **[ultralytics](https://github.com/ultralytics/ultralytics)** | YOLOv8 detection | De-facto SOTA detector |
| **[opencv-python](https://github.com/opencv/opencv)** | Computer vision primitives | Industry standard |
| **[supervision](https://github.com/roboflow/supervision)** | PolygonZone, annotators, line counters | Roboflow, ⭐ 8k+ |
| **[streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)** | Interactive polygon drawing in UI | The de-facto canvas component |
| **[mediapipe](https://github.com/google/mediapipe)** | Pose estimation (hand-near-shelf) | Google, ⭐ 30k+ |
| **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** | IsolationForest anomaly gating | Industry standard ML |
| **[filterpy](https://github.com/rlabbe/filterpy)** | Kalman filtering (optional) | Reference Kalman impl |
| **[pydantic](https://github.com/pydantic/pydantic)** | Typed config validation | Standard for Python config |
| **[loguru](https://github.com/Delgan/loguru)** | Logging | Drop-in upgrade |
| **[watchdog](https://github.com/gorakhargosh/watchdog)** | File/config hot-reload | Industry standard |

---

## 🚀 Quick Start

### 1. Install

```bash
# All dependencies (recommended)
pip install -r requirements.txt

# Or minimal (without pose/MediaPipe/anomaly ML)
pip install ultralytics opencv-python numpy streamlit pydantic pyyaml
```

### 2. Run the Streamlit app

```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

### 3. Run via CLI

```bash
# Process a local video
python main.py --video data/demo.mp4 --show

# RTSP camera
python main.py --rtsp rtsp://192.168.1.100:554/stream1 --show

# Disable new suspicious-activity detection
python main.py --video data/demo.mp4 --show --no-suspicious

# Enable pose-based detection
python main.py --video data/demo.mp4 --show

# Detect persons + bags
python main.py --video data/demo.mp4 --show --detect-classes 0,24,26,28
```

---

## 🗺️ Zone Editor — Draw Polygons on Video

1. Upload a video in the **📹 Live** tab.
2. Switch to the **🗺️ Zone Editor** tab.
3. Pick a tool (`polygon` or `rect`) and a default zone type.
4. Draw directly on the first frame. Use these colors for auto-classification:
   - 🔴 `#ff0000` → **restricted** (e.g. backroom, emergency exit)
   - 🟢 `#00ff00` → **checkout**
   - 🟡 `#ffff00` → **shelf**
   - 🔵 `#00ffff` → **entrance**
   - ⚪ `#ffffff` → **general**
5. Click **💾 Save Zones** — saved to `config/zones_runtime/{camera_id}.yaml`.

Zones are loaded automatically on the next pipeline run for the matching camera ID.

---

## ⚙️ Configuration

### `config/suspicious_activity.yaml`

All detection thresholds live here. Edit and restart, or use the in-app **Threshold Tuner** to override at runtime.

```yaml
detectors:
  loitering:
    enabled: true
    default_dwell_seconds: 120.0
    min_motion_variance: 0.05
    min_confidence: 0.6
    confirmation_frames: 10
    cooldown_seconds: 60.0

  restricted_zone:
    enabled: true
    default_severity: HIGH
    min_confidence: 0.85
    confirmation_frames: 3
    cooldown_seconds: 30.0
    enable_approach_escalation: true
  # ... etc

alerts:
  min_severity_to_notify: MEDIUM
  global_cooldown_seconds: 30.0
  min_confidence: 0.5
  dedup_window_seconds: 15.0
```

### `config/zones.yaml`

Polygon zones per camera. See **🗺️ Zone Editor** above to draw them visually.

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run only suspicious-activity tests
pytest tests/test_suspicious_activity.py -v

# Run anomaly-scorer training test (requires sklearn)
RUN_SKLEARN_TESTS=1 pytest tests/test_anomaly_scorer.py -v
```

The test suite includes **false-positive regression tests**:
- Active walking customers should NOT trigger loitering alerts
- Short tracks should be ignored
- Cooled-down alerts are suppressed
- Outside-zone tracks never trigger restricted-zone alerts

---

## 📊 Low-False-Positive Strategy

Each detector applies multiple guards before emitting an alert:

| Layer | Mechanism | Purpose |
|---|---|---|
| 1 | **Rule engine** | Fast, deterministic triggers for known patterns |
| 2 | **AnomalyScorer** (IsolationForest) | Suppress FPs that match rule shape but are normal in motion space |
| 3 | **Multi-frame confirmation** | Require N consecutive frames (configurable) |
| 4 | **Hysteresis** | Different enter/exit thresholds prevent flapping |
| 5 | **Per-rule cooldowns** | Independent of `AlertManager.rate_limit` |
| 6 | **Track quality filter** | Drop short / low-confidence tracks |
| 7 | **Pose corroboration** | Optional second signal from MediaPipe |
| 8 | **Evidence-signature dedup** | Suppress near-identical alerts inside a window |

---

## 📁 Project Structure

```
SmartStoreVisionAI/
├── main.py                         # CLI entry point (extended)
├── streamlit_app.py                # Streamlit UI (rewritten with tabs)
├── requirements.txt
├── config/
│   ├── settings.yaml               # General system config
│   ├── zones.yaml                  # Main polygon zones
│   ├── suspicious_activity.yaml    # NEW: thresholds for all detectors
│   ├── zones_runtime/              # NEW: per-camera drawn zones
│   └── cameras.yaml
├── dashboard/
│   ├── __init__.py
│   ├── polygon_editor.py           # NEW: streamlit-drawable-canvas
│   ├── alerts_panel.py             # NEW: live alert feed
│   └── threshold_tuner.py          # NEW: live tuning UI
├── src/
│   ├── detector/yolo_detector.py
│   ├── tracker/bytetrack.py
│   ├── analytics/
│   │   ├── zone_analytics.py
│   │   ├── heatmap.py
│   │   ├── behavior_detector.py    # backwards-compat wrapper
│   │   ├── movement_features.py    # NEW
│   │   ├── activity_rules.py       # NEW: YAML rule engine
│   │   ├── pose_analyzer.py        # NEW: MediaPipe
│   │   ├── anomaly_scorer.py       # NEW: IsolationForest
│   │   ├── multi_camera.py         # NEW: Re-ID
│   │   ├── supervision_zones.py    # NEW: supervision wrapper
│   │   └── suspicious_activity.py  # NEW: main orchestrator
│   ├── alerts/alert_manager.py     # extended w/ confidence+evidence
│   └── utils/
│       ├── config.py
│       ├── config_schema.py        # NEW: pydantic models
│       ├── helpers.py
│       └── zone_io.py              # NEW: save/load + canvas
└── tests/
    ├── fixtures/
    │   ├── __init__.py
    │   └── synthetic_tracks.py     # NEW: reproducible track generators
    ├── test_suspicious_activity.py # NEW
    ├── test_anomaly_scorer.py      # NEW
    ├── test_movement_features.py   # NEW
    ├── test_zone_io.py             # NEW
    └── test_detector.py
```

---

## 🎓 Custom Rules (YAML-driven)

Add your own rules to `config/suspicious_activity.yaml`:

```yaml
custom_rules:
  - name: "high_value_zone_loiter"
    when:
      zone_name_contains: "high_value"
      dwell_seconds: ">120"
      features.motion_var: "<0.05"
    then:
      activity: "loitering"
      severity: "HIGH"
      confidence: 0.8
      message: "Loitering near high-value display"

  - name: "evening_run"
    when:
      features.peak_speed: ">200"
      features.heading_change_count: "<3"
    then:
      activity: "running"
      severity: "MEDIUM"
      confidence: 0.7
```

Supported context keys: `track_id`, `zone_id`, `zone_type`, `zone_name`,
`dwell_seconds`, `occupancy`, `features.{any movement feature}`.

Comparison operators in conditions: `>`, `>=`, `<`, `<=`, `==`, `!=`. For
strings, `key_contains: "regex"` and `key_regex: "pattern"` are also available.

---

## ⚠️ Known Limitations

- **No object/bag detection** in YOLO person-only mode — `UnattendedCheckout` uses a stationary-pixel proxy. Enable with `--detect-classes 0,24,26,28` for full bag detection.
- **MediaPipe Pose** works best on a single person; crowded scenes may degrade.
- **AnomalyScorer** needs a one-time training step on **normal** traffic. Cold-start uses rule-only mode until you call `scorer.train(...)`.
- **Multi-camera Re-ID** via HSV histogram is weak in similar-clothing scenarios. The OSNet deep feature path is optional and requires `torch`.
- **`streamlit-drawable-canvas`** requires JavaScript enabled in the browser.

---

## 🏎️ Performance

| Hardware | Model | FPS (approx) |
|----------|-------|--------------|
| RTX 3080 | YOLOv8s | ~100 |
| CPU | YOLOv8n | ~15-20 |
| Jetson Orin | YOLOv8n | ~30+ |

Suspicious-activity detection adds <5ms per frame overhead on top of detection + tracking when pose/anomaly-scorer are disabled.

---

## 📜 License

MIT
