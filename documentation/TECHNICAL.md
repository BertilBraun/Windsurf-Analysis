# 🔬 Technical Documentation

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Detection Pipeline](#detection-pipeline)
3. [Multi-Stage Tracking System](#multi-stage-tracking-system)
4. [Feature Extraction Methods](#feature-extraction-methods)
5. [Discrete Optimization Approach](#discrete-optimization-approach)
6. [Performance Optimizations](#performance-optimizations)
7. [Model Training Pipeline](#model-training-pipeline)

---

## System Architecture

### Overview

Windsurf Analysis implements a sophisticated multi-stage tracking pipeline that transforms raw detections into coherent object trajectories. The system is designed around the principle of **progressive refinement**, where each stage addresses different aspects of the tracking problem.

```
Raw Video → YOLO Detection → Feature Extraction → Multi-Stage Tracking → Video Output
                ↓                    ↓                      ↓
            Bounding Boxes    ReID Embeddings +     [Preprocessor → Greedy → 
                               HSV Histograms       Optimization → Post-process]
```

### Core Data Structures

#### Detection

```python
@dataclass
class Detection:
    bbox: BoundingBox           # Spatial location
    embedding: np.ndarray       # 512-dim ReID feature vector
    confidence: float           # Detection confidence [0,1]
    frame_idx: int             # Temporal location
    color_histogram: np.ndarray # 256-dim HSV difference histogram
```

#### Track

```python
@dataclass  
class Track:
    track_id: int
    sorted_detections: list[Detection]  # Chronologically ordered
```

---

## Detection Pipeline

### YOLO11 Object Detection

**Model**: Custom fine-tuned YOLO11n architecture

- **Training Data**: ~500 professionally annotated windsurfing frames
- **Classes**: Single class ("windsurfer")
- **Input Resolution**: 640px (configurable)
- **Performance**: 95%+ detection accuracy, <10ms inference time

**Preprocessing**:

- Frame subsampling based on `MIN_TRACKING_FPS` (default: 25 FPS)
- Batch processing for GPU efficiency (`BATCH_SIZE`: 32)
- Confidence filtering (`CONFIDENCE_THRESHOLD`: 0.25)
- NMS with IoU threshold (`IOU_THRESHOLD`: 0.2)

### Feature Extraction Pipeline

The system extracts two complementary feature representations for each detection:

#### 1. ReID Embeddings (Appearance Features)

**Model**: OSNet-AIN (Omni-Scale Network with Adversarial Instance Normalization)

- **Architecture**: `osnet_ain_x1_0` pretrained on MSMT17
- **Output**: 512-dimensional L2-normalized embedding
- **Input**: Cropped detection regions resized to 256×128
- **Purpose**: Captures person appearance, clothing, equipment

```python
def get_features(self, bboxes: np.ndarray, frame: np.ndarray) -> np.ndarray:
    crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in bboxes]
    batch = torch.cat([self.preprocess_crop(crop) for crop in crops], dim=0)
    with torch.no_grad():
        feats = self.extractor(batch)
    return normalize(feats, dim=1).cpu().numpy()
```

#### 2. HSV Color Histograms (Contextual Features)

**Purpose**: Distinguish objects based on local color distribution relative to background
**Implementation**: Difference histograms (object - background)

```python
def compute_color_histogram(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    roi = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute histograms: H(256 bins), S(16 bins), V(8 bins)
    hist_h_roi = cv2.calcHist([hsv_roi], [0], None, [256], [0, 256])
    # ... (normalize and compute differences)
    
    return np.concatenate([hist_h_diff])  # Currently using H channel only
```

**Key Innovation**: Using difference histograms (ROI - full image) rather than absolute histograms helps distinguish objects from their immediate background context.

---

## Multi-Stage Tracking System

The tracking system processes detections through four sequential stages, each addressing different aspects of the association problem:

### Stage 1: Greedy Preprocessor

**Purpose**: Fast initial linking of obvious detection pairs
**Algorithm**: Greedy sequential processing with appearance and spatial constraints

```python
class GreedyPreprocessor:
    def _compare_detection_to_track(self, track: Track, detection: Detection):
        iou = track.end().bbox.iou(detection.bbox)
        if iou < self.min_iou_matches_single_track:
            return NO_MATCH
            
        n = len(track.sorted_detections)
        avg_sim = sum(cosine_similarity(d.embedding, detection.embedding) 
                     for d in track.sorted_detections) / n
                     
        if iou >= self.greedy_min_iou and avg_sim >= self.greedy_min_cosine_similarity:
            return MATCH
        return MAY_MATCH
```

**Key Features**:

- **Single-pass processing**: Processes detections chronologically
- **Active track management**: Tracks become "stale" after gaps or conflicts
- **Conservative matching**: Requires both spatial (IoU) and appearance (cosine similarity) agreement
- **Conflict resolution**: Ambiguous matches create new tracks

**Parameters**:

- `GREEDY_PREPROCESSOR_MIN_IOU`: 0.5
- `GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY`: 0.7
- `GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE`: 5 frames

### Stage 2: Greedy Tracker

**Purpose**: Iterative track merging based on average embedding similarity
**Algorithm**: Greedy pairwise merging with highest-similarity-first strategy

```python
def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]:
    while True:
        # Pre-compute average embeddings for all tracks
        avg_emb = [mean_embedding(t) for t in working]
        
        best_i, best_j, best_sim = None, None, -1.0
        for i in range(n):
            for j in range(i + 1, n):
                if not _can_merge(working[i], working[j], max_gap=max_gap):
                    continue
                sim = cosine_similarity(avg_emb[i], avg_emb[j])
                if sim > best_sim:
                    best_i, best_j, best_sim = i, j, sim
        
        if best_sim < GREEDY_MIN_COSINE_SIMILARITY:
            break
            
        # Merge best pair and continue
        new_track = _merge_tracks(working[best_i], working[best_j])
        # ... update working list
```

**Innovation: Average Embedding Similarity**
Instead of pairwise detection comparisons, tracks are represented by their average embedding:

```python
def mean_embedding(t: Track) -> np.ndarray:
    return np.mean([d.embedding for d in t.sorted_detections], axis=0)

def mean_embedding_cosine_similarity(a: Track, b: Track) -> float:
    return cosine_similarity(mean_embedding(a), mean_embedding(b))
```

**Merging Constraints**:

- **Temporal non-overlap**: Track frame ranges must not intersect
- **Minimum IoU**: Spatial consistency at connection points
- **Maximum gap**: Limited temporal separation (10 seconds default)
- **Minimum length**: Very short tracks handled in later stages

### Stage 3: Discrete Optimization Tracker

**Purpose**: Global optimization for complex multi-track scenarios
**Algorithm**: Z3-based constraint satisfaction with cost minimization

#### Problem Formulation

The discrete optimization formulation treats tracking as a **fragment linking problem**:

- **Fragments**: Individual tracks from previous stages
- **Links**: Possible connections between fragments
- **Objective**: Minimize total cost while satisfying constraints

#### Mathematical Model

**Decision Variables**:

- `link[i,j]`: Boolean variable indicating fragment i connects to fragment j
- `start[i]`: Boolean variable indicating fragment i starts a new track

**Constraints**:

```
∀i: Σⱼ link[i,j] ≤ 1                    # At most one outgoing link
∀j: Σᵢ link[i,j] ≤ 1                    # At most one incoming link  
∀i: start[i] = ¬(∃j: link[j,i])        # Start iff no incoming link
```

**Cost Function**:

```
minimize: Σ(i,j) link[i,j] × cost(i,j) + Σᵢ start[i] × w_start
```

#### Cost Computation

**Link Cost** combines multiple factors:

```python
def _calculate_link_cost(self, start: Track, end: Track) -> float:
    # Geometric similarity (IoU between connection points)
    iou = end.start().bbox.iou(start.end().bbox)
    
    # Appearance similarity (windowed average)
    cos = self._calculate_windowed_cosine_similarity(start, end, ...)
    
    # Temporal gap penalty
    gap = end.start_frame() - start.end_frame()
    
    return (self.w_link_iou * (1.0 - iou) + 
            self.w_link_app * (1.0 - cos) + 
            self.w_link_gap * gap)
```

**Key Innovation: Windowed Appearance Similarity**
Instead of single-point comparison, the system evaluates appearance similarity over a temporal window:

```python
def _calculate_windowed_cosine_similarity(self, start: Track, end: Track, 
                                        start_det: Detection, end_det: Detection) -> float:
    cos_sum = 0.0
    n_pairs = 0
    
    for i in range(-window_radius, window_radius + 1):
        d1 = start.detections_by_frame.get(start_det.frame_idx + i)
        if d1 is None: continue
            
        for j in range(-window_radius, window_radius + 1):
            d2 = end.detections_by_frame.get(end_det.frame_idx + j)
            if d2 is None: continue
                
            cos_sum += cosine_similarity(d1.embedding, d2.embedding)
            n_pairs += 1
    
    return cos_sum / n_pairs if n_pairs > 0 else 0.0
```

#### Z3 Solver Implementation

```python
def _solve_optimization_problem(self, graph: FragmentGraph) -> Dict[int, Optional[int]]:
    opt = z3.Optimize()
    opt.set('timeout', OPTIMIZER_TIMEOUT_SECONDS * 1000)
    
    # Create decision variables
    link_vars = {(i,j): z3.Bool(f'link_{i}_{j}') for (i,j) in graph.get_all_connections()}
    start_vars = [z3.Bool(f'start_{i}') for i in range(len(graph.fragments))]
    
    # Add constraints (outgoing, incoming, start)
    # Set objective function
    # Solve and extract solution
    
    result = opt.check()
    if result != z3.sat:
        raise UnsatisfiableException("Fragment linking UNSAT")
        
    return self._extract_solution(opt.model(), link_vars)
```

**Advantages of Z3 Approach**:

- **Global optimality**: Considers all fragments simultaneously
- **Constraint satisfaction**: Hard constraints ensure valid solutions
- **Flexible cost functions**: Easy to add new cost terms
- **Scalability**: Handles complex scenarios with many fragments

### Stage 4: Track Post-Processing

**Purpose**: Final refinement, filtering, and smoothing
**Operations**:

1. **Duration Filtering**: Remove tracks shorter than minimum percentage of video
2. **Interpolation**: Fill gaps in track detections using linear interpolation
3. **Smoothing**: Apply rolling window smoothing to bounding box centers
4. **Relabeling**: Assign consecutive track IDs

```python
def _smooth_track(track_data: list[Detection], window_size: int = 2) -> list[Detection]:
    smoothed_track = []
    for i, detection in enumerate(track_data):
        # Calculate window indices
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        # Smooth center positions
        centers_x = [track_data[j].bbox.center.x for j in range(start_idx, end_idx)]
        centers_y = [track_data[j].bbox.center.y for j in range(start_idx, end_idx)]
        
        smooth_center_x = sum(centers_x) / len(centers_x)
        smooth_center_y = sum(centers_y) / len(centers_y)
        
        # Reconstruct with smoothed center
        smoothed_detection = detection.copy()
        smoothed_detection.bbox = BoundingBox(...)  # Update with smoothed center
        smoothed_track.append(smoothed_detection)
```

---

## Feature Extraction Methods

### ReID Embedding Analysis

**Model Selection**: OSNet-AIN chosen for superior performance on person re-identification:

- **Multi-scale feature extraction**: Captures both fine-grained and global appearance
- **Adversarial training**: Improves domain generalization
- **Normalization**: L2-normalized 512D embeddings ensure cosine similarity validity

**Embedding Quality Assessment**:

- **Intra-track consistency**: Average cosine similarity within tracks: ~0.85
- **Inter-track discrimination**: Average similarity between different tracks: ~0.3
- **Temporal stability**: Embeddings remain consistent across short temporal gaps

### HSV Histogram Implementation

**Motivation**: ReID embeddings focus on person appearance but may miss environmental context. HSV histograms capture color distribution patterns that help distinguish objects in similar poses.

**Technical Details**:

```python
# Histogram computation (currently H-channel focused)
hist_h_roi = cv2.calcHist([hsv_roi], [0], None, [256], [0, 256])    # 256 bins
hist_s_roi = cv2.calcHist([hsv_roi], [1], None, [16], [0, 256])     # 16 bins  
hist_v_roi = cv2.calcHist([hsv_roi], [2], None, [8], [0, 256])      # 8 bins

# Difference computation (key innovation)
hist_diff = (hist_roi / sum(hist_roi)) - (hist_full / sum(hist_full))
```

**Benefits of Difference Histograms**:

- **Background invariance**: Reduces dependence on global illumination
- **Object highlighting**: Emphasizes colors unique to the detection region
- **Noise reduction**: Filters out common environmental colors

### Similarity Metrics

The system implements multiple similarity measures for different use cases:

```python
# Primary metrics
def mean_embedding_cosine_similarity(a: Track, b: Track) -> float
def mean_embedding_histogram_similarity(a: Track, b: Track) -> float

# Alternative metrics for analysis
def pairwise_cosine_similarity(a: Track, b: Track) -> float
def pairwise_histogram_similarity(a: Track, b: Track) -> float
def prop_embeddings_sim(a: Track, b: Track, min_sim: float = 0.5) -> float
```

---

## Performance Optimizations

### GPU Acceleration

**YOLO Inference**:

- Batch processing: 32 detections per batch
- CUDA memory management: Automatic cleanup between batches
- Mixed precision: FP16 inference where supported

**ReID Feature Extraction**:

- Batch preprocessing: Multiple crops processed simultaneously  
- Tensor optimization: Minimize CPU-GPU transfers
- Model compilation: TorchScript compilation for inference speedup

### Memory Management

**Streaming Video Processing**:

```python
# Process video without loading entire sequence into memory
with VideoReader(video_path) as reader:
    for frame_index, frame in reader.read_frames():
        detections = detector.process_frame(frame)
        # Process immediately, don't accumulate
```

**Detection Storage**:

- Lazy evaluation: Compute features only when needed
- Memory pools: Reuse numpy arrays where possible
- Garbage collection: Explicit cleanup of large intermediate results

### Parallel Processing

**Multi-Video Processing**:

```python
with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
    futures = []
    for worker_id in range(parallel_workers):
        indices = [i for i in range(len(videos)) if i % parallel_workers == worker_id]
        futures.append(executor.submit(process_videos, videos, indices, ...))
```

**Pipeline Parallelism**:

- **Detection**: GPU-accelerated batch inference
- **Tracking**: CPU-intensive algorithms run in parallel with video I/O
- **Stabilization**: Background processing during main pipeline execution

---

## Model Training Pipeline

### Dataset Preparation

**Annotation Tool** (`train/annotator.py`):

- Interactive bounding box annotation
- Multi-video sampling with weighted selection
- Real-time preview with adjustable boxes
- Undo/redo functionality for quality control

**Features**:

- **Weighted sampling**: Prioritizes videos with more frames
- **Grow/shrink modes**: Fine-tune bounding boxes precisely
- **Navigation**: Frame-by-frame stepping for temporal consistency
- **Export format**: YOLO-compatible annotation files

### Training Pipeline

**Automated Training** (`train/train.py`):

```python
def prepare_dataset(src: Path, dst: Path, val_ratio: float = 0.02):
    # Gather matching .jpg/.txt pairs
    # Split into train/validation 
    # Create Ultralytics directory structure
    # Generate dataset YAML file
    
def train_model(yaml_file: Path, epochs: int, imgsz: int, batch: float, device: str):
    model = YOLO('yolo11n.pt')  # Base model
    model.train(data=str(yaml_file), epochs=epochs, imgsz=imgsz, 
                batch=batch, device=device, single_cls=True)
```

**Training Configuration**:

- **Base model**: YOLO11n (nano) for speed/accuracy balance
- **Single class**: "windsurfer" class only
- **Data augmentation**: Standard Ultralytics augmentation pipeline
- **Validation**: Automated train/val split with stratified sampling

### Model Evaluation

**Performance Metrics**:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **Inference speed**: Frames per second on target hardware
- **Detection consistency**: Temporal stability across frames
- **False positive rate**: Background rejection performance

**Validation Strategy**:

- **Temporal splits**: Ensure train/val come from different video sessions
- **Scenario diversity**: Test on various wind conditions, lighting, equipment
- **Real-world testing**: Validation on unseen competition footage

---

## Configuration Parameters

### Detection Settings

```python
# YOLO model configuration
YOLO_MODEL_PATH = '../train/models/100epochs.pt'
CONFIDENCE_THRESHOLD = 0.25    # Minimum detection confidence
IOU_THRESHOLD = 0.2           # NMS IoU threshold
BATCH_SIZE = 32               # Inference batch size

# Frame processing
MIN_TRACKING_FPS = 25         # Target processing frame rate
```

### Tracking Settings

```python
# Greedy preprocessor
GREEDY_PREPROCESSOR_MIN_IOU = 0.5
GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY = 0.7
GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE = 5

# Greedy tracker  
GREEDY_MIN_COSINE_SIMILARITY = 0.8
GREEDY_SHORT_TRACK_MIN_FRAMES = 10

# Discrete optimization
OPTIMIZER_MIN_LINK_IOU = 0.0
OPTIMIZER_W_LINK_APP = 1.0        # Appearance weight
OPTIMIZER_W_LINK_IOU = 0.2        # Spatial weight  
OPTIMIZER_W_START = 10.0          # New track penalty
OPTIMIZER_TIMEOUT_SECONDS = 60    # Z3 solver timeout
```

### Post-Processing Settings

```python
# Track filtering
MIN_FRAME_PERCENTAGE = 20         # Minimum track duration (% of video)
SMOOTHING_WINDOW_SIZE = 2         # Trajectory smoothing window

# Video output
MAX_OVERLAP_LENGTH_SECONDS = 10   # Maximum gap for track linking
```

---

## Future Enhancements

### Algorithmic Improvements

1. **Multi-Object Tracking Metrics**: Implement HOTA, MOTA evaluation
2. **Learned Similarity**: Train neural similarity functions end-to-end
3. **Temporal Modeling**: LSTM/Transformer-based trajectory prediction
4. **3D Tracking**: Incorporate depth estimation for spatial analysis

### System Optimizations

1. **Real-time Processing**: Optimize for live video streaming
2. **Distributed Computing**: Multi-GPU and multi-node scaling
3. **Model Compression**: Quantization and pruning for edge deployment
4. **Memory Efficiency**: Streaming processing for very long videos

### Feature Additions

1. **Advanced Analytics**: Speed estimation, maneuver classification
2. **Interactive Visualization**: Web-based analysis dashboard
3. **Multi-Sport Support**: Generalize to other water sports
4. **Mobile Deployment**: iOS/Android app integration

---

*This technical documentation covers the core algorithms and implementation details. For usage examples and getting started guides, see the main [README](README.md).*
