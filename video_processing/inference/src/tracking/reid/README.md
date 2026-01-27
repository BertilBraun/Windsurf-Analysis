This folder provides various implementations for Person/Object Re-Identification (ReID), used to extract feature embeddings from image crops for tracking across frames.

### Core Interface
*   **ReID Protocol**: Defines the standard interface `get_features_for_crops(crops: List[np.ndarray]) -> Sequence[Embedding]`.

### Implementations

#### Deep Learning Models
*   **ReIDOSNet**: Uses the OSNet architecture (specifically `osnet_ain_x1_0` via `torchreid`) to extract deep visual features. Supports CUDA, MPS, and CPU execution with FP16 support on CUDA.
*   **ReIDViT**: Uses Vision Transformers (via `open_clip`, specifically `ViT-B-32`) to generate embeddings.

#### Color & Histogram Methods
*   **ReIDColorABStripeHistogram**: A sophisticated color-based descriptor that:
    *   Partitions crops into vertical stripes (default 3) plus a global block.
    *   Computes joint Lab (a,b) and circular Hue histograms.
    *   Applies saturation-based pixel weighting ($S^\gamma$).
    *   Includes logic for foreground/background masking based on border statistics and Mahalanobis distance.
    *   Returns Hellinger distance-compatible embeddings.
*   **ReIDColorHistogram**: A simple 3D Lab color histogram implementation. It applies a 15% center-crop margin to focus on the subject and reduce background noise.

### TODO
*   Refine the foreground masking logic in `ReIDColorABStripeHistogram` (currently uses a fixed 10% margin fallback in the active `_compute_mask` method).
*   Standardize visualization tools across different ReID methods.
*   Implement batch processing for histogram-based methods to match deep learning performance patterns.
