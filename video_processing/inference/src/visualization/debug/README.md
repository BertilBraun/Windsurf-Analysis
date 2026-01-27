Interactive debugging and visualization tools for tracking internals, Kalman filter states, and track fragment linking.

### Core Components

*   **draw.py**: Low-level OpenCV wrappers for drawing primitives.
    *   `draw_bounding_box` / `draw_text` / `draw_arrow`: Standardized annotation helpers.
    *   `new_canvas`: Utility to create empty black images.
    *   `compose_side_by_side`: Concatenates images horizontally with automatic height normalization.
    *   `draw_heatmap`: Visualizes cost matrices (e.g., assignment costs) with row/column labels, colorbar, and numeric cell values.
*   **graph.py**: Interactive fragment graph using Matplotlib and NetworkX.
    *   `EdgeRecord`: Dataclass storing motion, appearance, and gap costs between fragments.
    *   `show_graph_interactive`: Visualizes track fragments as nodes positioned by time. Supports click callbacks on nodes (for bounding boxes) and edges (for cost comparisons).
*   **overlays.py**: Rendering logic for tracking metadata.
    *   `DetectionsOverlay`: Visualizes raw detection boxes.
    *   `KalmanOverlay`: Visualizes predicted bounding boxes and velocity vectors.
    *   `CameraMotionTrailOverlay`: Visualizes camera movement history as a colored trail.
    *   `compose_fragment_pair_view`: Specialized side-by-side comparison of two fragments with a metrics banner (Mahalanobis distance, NLL costs).
*   **session.py**: Unified interface for debug lifecycle management.
    *   `DebugSession`: Protocol defining frame display, HUD management, and user input handling.
    *   `Cv2DebugSession`: Active implementation managing frame buffers and window lifecycles. Supports loading frames directly from video files.
    *   `NullDebugSession`: No-op implementation for production environments.
*   **viewer.py**: OpenCV window management.
    *   Handles HUD (Heads-Up Display) text rendering.
    *   Supports mouse callback registration for interactive windows.
    *   `scroll`: Provides keyboard-based frame navigation (Arrows for stepping, `,` / `.` for 30-frame jumps, `[` / `]` for custom time jumps).

### TODO

*   Implement current frame transform fetching in `CameraMotionTrailOverlay.apply`.
*   Standardize frame index tracking within `Overlay` objects.
