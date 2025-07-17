# 🏄‍♂️ Windsurf Analysis: Professional AI-Powered Windsurfing Video Intelligence

**Enterprise-grade computer vision system for comprehensive windsurfing video analysis, tracking, and performance optimization.**

Windsurf Analysis is a state-of-the-art AI system that combines custom-trained YOLO models, advanced multi-object tracking algorithms, and professional video processing pipelines to automatically detect, track, and analyze windsurfers in complex marine environments. The system provides end-to-end processing from raw footage to stabilized individual clips with detailed performance analytics.

- **Technique Analysis**: Review your own technique with frame-by-frame analysis - analyze successful and unsuccessful maneuvers and identify areas for improvement
- **Progress Tracking**: Compare improvement over time with consistent metrics
- **Efficient Review**: Quickly isolate specific maneuvers or problem areas
- **Demonstration Tools**: Create annotated examples for teaching

![example](documentation/processed.gif)

Example of the usage of the system: Input is a raw video of a windsurf session (top left), we process it using AI to detect the surfer and track their movements (bottom left), we then extract the individual surfer videos, making sure, that the surfer is always in the center of the frame and stabilize their clips (right).

## 🔬 Core Technologies

### 🧠 **Custom YOLO Fine-Tuning Pipeline**

- **Domain-specific training**: Custom YOLO11 models fine-tuned on comprehensive windsurfing datasets
- **Ultralytics integration**: Professional training infrastructure with configurable hyperparameters
- **Performance optimization**: Batch processing, GPU acceleration, and inference optimization
- **Model versioning**: Systematic model management and deployment pipeline

### 🎯 **Advanced Multi-Object Tracking**

- **BoT-SORT integration**: State-of-the-art tracking with ReID features and appearance modeling
- **Intelligent track merging**: Spatial-temporal analysis with histogram-based appearance matching
- **Robust identity preservation**: Handles occlusions, rapid movements, and challenging marine conditions
- **Configurable tracking parameters**: Fine-tuned for windsurfing-specific movement patterns

### 🎬 **Professional Video Processing Pipeline**

- **AABB-based stabilization**: Axis-aligned bounding box tracking for motion-aware stabilization
- **FFmpeg integration**: Professional-grade video encoding with configurable quality parameters
- **Automated clip extraction**: Intelligent scene segmentation based on tracking continuity
- **Batch processing**: Parallel video processing with worker pool architecture

## ⚙️ Technical Architecture

### **Machine Learning Infrastructure**

- **YOLO11 Base Models**: Fine-tuned on ~500 annotated windsurfing frames with diverse conditions
- **Custom Dataset Pipeline**: Automated train/validation splits with data augmentation
- **PyTorch Integration**: GPU-accelerated inference with CUDA optimization
- **Model Performance**: 95%+ detection accuracy, sub-10ms inference time per frame

### **Multi-Object Tracking System**

- **BoT-SORT Algorithm**: Advanced tracking with ReID features and global motion compensation
- **Appearance Modeling**: HSV histogram-based similarity matching for track association
- **Temporal Consistency**: Intelligent track merging using spatial-temporal distance metrics
- **Robust Re-identification**: Handles track fragmentation with configurable similarity thresholds

### **Video Processing Pipeline**

- **FFmpeg Backend**: Professional video encoding with H.264/H.265 support
- **Stabilization Engine**: Two-pass stabilization using vidstab filters with motion detection
- **Frame Management**: Efficient memory usage with streaming video processing

### **Track Processing Algorithms**

- **Smoothing Filters**: Configurable window-based trajectory smoothing
- **Track Validation**: Minimum frame percentage and duration filtering
- **Spatial Analysis**: AABB-based collision detection and overlap handling

### **Performance Optimization**

- **Parallel Processing**: Multi-threaded worker pools for CPU-intensive operations
- **Batch Inference**: Optimized YOLO batch processing for improved throughput
- **Memory Management**: Streaming video processing to handle large files efficiently
- **GPU Utilization**: CUDA acceleration for all compatible operations

## 📈 Model Training & Validation

For more of the training results, check out the [documentation/train_results](documentation/train_results) folder.

### **Dataset Specifications**

- **Training Dataset**: ~500 professionally annotated frames across diverse conditions
- **Athlete Diversity**: Multiple skill levels from recreational to professional competition
- **Scenario Coverage**: Jibes, rides, transitions, crashes, and multi-surfer interactions
- **Environmental Conditions**: Various wind conditions, lighting scenarios, and water states
- **Equipment Variations**: Different board types, sail configurations, and rigging setups
- **Camera Perspectives**: Multiple angles, distances, and stabilization conditions

### **Training Infrastructure**

- **Base Model**: YOLO11n architecture with transfer learning
- **Training Parameters**: Configurable epochs, batch size, image resolution (640px default)
- **Validation Split**: Automated validation set with stratified sampling
- **Data Augmentation**: Standard Ultralytics augmentation pipeline
- **Hardware Requirements**: CUDA-compatible GPU for training acceleration

## 🚀 System Requirements & Installation

### **Software Dependencies**

- **Python**: 3.10+ with pip package management
- **CUDA**: 11.8+ for GPU acceleration (optional)
- **FFmpeg**: 4.4+ with vidstab plugin for stabilization
- **PyTorch**: 2.0+ with CUDA support

### **Installation & Setup**

```bash
# Clone repository
git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### **Usage Examples**

#### **Basic Video Analysis**

```bash
# Process single video with annotations
python src/main.py "path/to/session.mp4" --draw-annotations

# Batch process multiple videos
python src/main.py "videos/*.mp4"

# Custom output directory
python src/main.py "footage.mp4" --output-dir analysis/ --draw-annotations
```

#### **Model Training & Fine-tuning**

```bash
# Train custom model on your dataset
python train/train.py \
    --src ./train/dataset \
    --dst ./train/datasets/custom \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device 0
```

#### **Advanced Configuration**

```bash
# Modify detection thresholds in src/settings.py
IOU_THRESHOLD = 0.2          # Intersection over Union threshold
CONFIDENCE_THRESHOLD = 0.1   # Detection confidence threshold
MIN_TRACKING_FPS = 25        # Minimum tracking frame rate

# Adjust tracking parameters
MAX_TEMPORAL_DISTANCE_SECONDS = 10.0    # Track merging time window
MAX_SPATIAL_DISTANCE_BB = 1.5           # Spatial distance for track linking
HISTOGRAM_SIMILARITY_THRESHOLD = 0.9     # Appearance similarity threshold
```

## 📁 Output Structure & Data Flow

### **Generated Files**

```text
output_directory/
├── <input_file_name1>+00_annotated.mp4     # Annotated video with tracking trails
├── <input_file_name1>+01.stabilized.mp4    # Stabilized individual video
├── <input_file_name1>+01.start_time.json   # Start time of the individual video
├── <input_file_name1>+02.stabilized.mp4    # Stabilized individual video
├── <input_file_name1>+02.start_time.json   # Start time of the individual video
|
├── <input_file_name2>+00_annotated.mp4     # Annotated video with tracking trails
├── <input_file_name2>+01.stabilized.mp4    # Stabilized individual video
├── <input_file_name2>+01.start_time.json   # Start time of the individual video
...
```
