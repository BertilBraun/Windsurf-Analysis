# 🏄‍♂️ Windsurf Analysis: AI-Powered Windsurfing Video Intelligence

**Transform your windsurfing footage into actionable insights with cutting-edge computer vision and multi-object tracking.**

Windsurf Analysis automatically detects, tracks, and extracts individual windsurfer videos from complex marine environments. Using custom YOLO models, advanced ReID features, and sophisticated multi-stage tracking algorithms, it delivers professional-grade analysis for technique improvement, progress tracking, and coaching.

![Demo](documentation/processed.gif)

*From raw session footage (top left) to AI-powered detection and tracking (bottom left) to stabilized individual videos (right)*

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

### Basic Usage

```bash
# Process a single video
python src/main.py "session.mp4" --draw-annotations

# Batch process multiple videos
python src/main.py "videos/*.mp4" --output-dir results/

# Custom configuration
python src/main.py "footage.mp4" --output-dir analysis/ --parallel-workers 4
```

### Output Structure

```
output_directory/
├── session+00_annotated.mp4        # Full video with tracking annotations
├── session+01.mp4                  # Individual surfer video #1  
├── session+01.stabilized.mp4       # Stabilized version
├── session+01.start_time.json      # Timing metadata
├── session+02.mp4                  # Individual surfer video #2
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

### 3. **Video Processing**

- Individual video extraction with intelligent cropping
- Professional video stabilization using FFmpeg
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

*Built with ❤️ for the windsurfing community*
