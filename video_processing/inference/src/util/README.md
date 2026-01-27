Utility functions for mathematical operations, caching, logging, and video handling.

### Algebra and Math
- Normalization: L1 and L2 normalization for numpy arrays.
- Distance Metrics: Implementations for Cosine similarity, Hellinger distance, and Chi-squared distance.
- Probability Mapping: Functions for Sigmoid, Platt scaling, Negative Log-Likelihood (NLL), and Log-odds costs.
- Basic Helpers: Linear interpolation (lerp), floor, and ceil operations.

### Similarity and Embeddings
- Embedding Protocol: Standard interface for distance, interpolation, and probability calculations between feature vectors.
- VectorEmbedding: L2-normalized vector handling using cosine similarity.
- HistogramEmbedding: L1-normalized histogram handling using Chi-squared distance.
- HellingerEmbedding: Specialized embedding for non-negative vectors using Hellinger distance.

### Video I/O
- VideoReader: Generator-based frame reading with support for skipping every Nth frame.
- VideoWriter: Context-managed wrapper for OpenCV's VideoWriter.
- LiveVideoStreamer: Real-time playback utility that maintains target FPS and handles window lifecycle.
- VideoInfo: Dataclass for storing FPS, dimensions, and total frame counts.

### Caching and Performance
- @cache_to_file: Decorator that persists function results to disk (under tmp/cache/) using MD5 hashes of arguments.
- timeit: Context manager for measuring and printing execution time of code blocks.

### Logging and Helpers
- setup_logging: Configures standard logging to both console and file.
- log_and_reraise: Utility to log exceptions with stack traces before re-raising them, useful for multi-process debugging.
