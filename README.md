# 🏄‍♂️ Windsurf Analysis: AI-Powered Windsurfing Video Intelligence

**Transform your windsurfing footage into actionable insights with cutting-edge computer vision and multi-object tracking.**

Windsurf Analysis automatically detects, tracks, and extracts individual windsurfer videos from complex marine environments. Using custom YOLO models, advanced ReID features, and sophisticated multi-stage tracking algorithms, it delivers professional-grade analysis for technique improvement, progress tracking, and coaching.

![Demo](documentation/processed.gif)

Note: From raw session footage (top left) to AI-powered detection and tracking (bottom left) to stabilized individual videos (right).

## ✨ Key Features

- **🎯 Precision Detection**: Custom YOLO11 models fine-tuned specifically for windsurfing scenarios
- **🧠 Smart Tracking**: Multi-stage tracking pipeline combining appearance modeling, spatial reasoning, and discrete optimization
- **🎬 Individual Videos**: Automatically extract and stabilize focused videos for each detected surfer
- **📊 Rich Analytics**: Detailed performance metrics, trajectory analysis, and technique insights
- **⚡ GPU Accelerated**: Optimized for speed with CUDA support and batch processing

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup (optional but recommended)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Player

After processing, launch the interactive Player to browse results:

```bash
python src/player_main.py  # prompts to select the output directory
# or pass a directory explicitly
python src/player_main.py "results/"
```

Controls:

- Space: Play/Pause
- Left/Right: Step frame
- Shift+Left/Right: -/+ 5 seconds
- Ctrl+Left/Right: -/+ 30 seconds
- `+` / `-`: Speed up/down
- Mouse wheel: Zoom (in overview); zoom-in keeps the pixel under the cursor stationary
- Mouse click: Enter detailed view for the surfer under the cursor
- Esc: Back to overview
- n / p: Next/Previous video in directory

### Basic Usage

```bash
# Process a single video (saves compact metadata for the Player)
python src/main.py "session.mp4" --output-dir results/

# Batch process multiple videos
python src/main.py "videos/*.mp4" --output-dir results/

# Custom configuration (parallel workers, debug views)
python src/main.py "footage.mp4" --output-dir analysis/ --parallel-workers 4 --debug-views

### CLI Flags (Processing)

- `--output-dir`: Directory to store outputs (metadata and optional videos)
- `--generate-videos`: Also render per-surfer videos (legacy export)
- `--draw-annotations`: Render a full annotated overview video
- `--stabilize`: Stabilize generated per-surfer videos
- `--parallel-workers`: Parallel processing degree
```

### Output Structure

By default, the processing step writes compact track metadata used by the interactive Player. Large video exports are optional.

```text
output_directory/
└── session.tracks.pkl              # Serialized metadata for the Player (pickle)
```

Optional legacy exports (only when requested):

```text
output_directory/
├── session+00_annotated.mp4        # Overview with all tracks (when --draw-annotations)
├── session+01.mp4                  # Individual surfer video #1  (when --generate-videos)
├── session+01.stabilized.mp4       # Stabilized version          (when --stabilize)
└── ...
```

## 🏗️ System Architecture

Windsurf Analysis uses a sophisticated multi-stage pipeline:

### 1. **AI Detection Pipeline**

- **YOLO11** object detection fine-tuned on windsurfing datasets
- **ReID Feature Extraction** using OSNet for robust appearance modeling  
- **HSV Color Histograms** for visual appearance discrimination

### 2. **Multi-Stage Tracking System**

- **Greedy Preprocessor**: Fast initial track linking with appearance and spatial cues
- **Discrete Optimization**: ILP-based global optimization for complex scenarios
- **Post-Processing**: Filtering, smoothing, and trajectory refinement

### 3. **Player-Centric Processing**

- Compact metadata export for interactive review
- Optional: annotated overview video and per-surfer clips
- Optional: video stabilization for exported clips
- Batch processing with parallel workers

## 📋 Requirements

### Software Dependencies

- **Python**: 3.10+
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **FFmpeg**: 4.4+ with vidstab plugin

### Hardware Recommendations

- **GPU**: NVIDIA GPU with 4GB+ VRAM (for optimal performance)
- **RAM**: 8GB+ system memory
- **Storage**: SSD recommended for large video processing

## ⚙️ Configuration

Key settings can be adjusted in `src/settings.py`:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.2

# Tracking parameters  
GREEDY_MIN_COSINE_SIMILARITY = 0.8
OPTIMIZER_MIN_LINK_IOU = 0.0
OPTIMIZER_W_LINK_APP = 1.0

# Processing settings
MIN_FRAME_PERCENTAGE = 20
BATCH_SIZE = 32

# Player detailed-view rendering
TARGET_BBOX_HEIGHT_RATIO = 0.5
SMOOTHING_ALPHA = 0.2
MIN_SCALE = 0.5
MAX_SCALE = 3.0
```

## 🎓 Training Custom Models

Train your own detection models on custom datasets:

```bash
# Prepare dataset from annotated frames
python train/train.py \
    --src ./dataset \
    --dst ./datasets/custom \
    --epochs 100 \
    --imgsz 640 \
    --batch 16
```

See the annotation tool in `train/annotator.py` for creating training data.

## 📊 Technical Deep Dive

For detailed information about the algorithms, architecture, and implementation:

**📖 [View Technical Documentation](documentation/TECHNICAL.md)**

Topics covered:

- Multi-stage tracking pipeline details
- Greedy vs. discrete optimization approaches  
- Feature extraction and embedding methods
- HSV histogram implementation
- Performance optimization strategies
- Model training and evaluation

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Algorithm improvements**: Better tracking heuristics, appearance models
- **Performance optimization**: GPU utilization, memory efficiency  
- **Dataset expansion**: More diverse training scenarios
- **Feature additions**: New analytics, export formats

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLO framework
- **TorchReID** for person re-identification models
- **VidStab** for video stabilization
- **OpenCV** community for computer vision tools

## 📧 Contact

For questions, support, or collaboration opportunities, please open an issue or reach out to the development team.

---

Built with ❤️ for the windsurfing community.

## 🌐 Web App (Production MVP)

This repo includes a minimal backend API (`backend/`) and a Modal GPU function (`modal_app/`) implementing the production requirements in `PRODUCTION_REQUIREMENTS.md`:

- FastAPI backend: Basic Auth, upload AC, job management, Modal webhook, S3 (R2) storage.
- Modal GPU function: downloads AC, runs existing pipeline, serializes tracks to results JSON, posts completion.
- See `DEPLOYMENT.md` for end-to-end setup with Neon (Postgres), Cloudflare R2 (S3), Modal, and a React frontend.
