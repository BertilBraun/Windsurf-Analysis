Tools for estimating global camera motion and tracking objects using a Kalman Filter with motion compensation.

### Components

*   **GMC (Global Motion Compensation)**: Estimates rigid transformations between consecutive frames.
    *   Uses `cv2.goodFeaturesToTrack` and Lucas-Kanade optical flow.
    *   Employs RANSAC (`estimateAffinePartial2D`) to find the best rigid fit.
    *   Supports image downscaling for performance and masking to exclude foreground objects.
*   **CMC (Camera Motion Compensation)**: Applies GMC transformations to Kalman Filter states.
    *   Adjusts position ($cx, cy$) and velocity ($vx, vy$) components of the state vector.
    *   Symmetrizes the covariance matrix after transformation to prevent numerical drift.
*   **Kalman Filter**: A constant-velocity linear filter for tracking bounding boxes.
    *   **State**: $[cx, cy, w, h, vx, vy, vw, vh]$ (center coordinates, dimensions, and velocities).
    *   **Joseph Update**: Uses the Joseph form covariance update for numerical stability and to ensure the matrix remains positive semi-definite.
    *   **Gating**: Supports Mahalanobis and Gaussian distance metrics for detection-to-track association.
    *   **Inflation**: Provides `display_bbox` which inflates the bounding box based on state uncertainty (Chi-squared quantiles).
    *   **KFState**: A dataclass wrapper that manages state history, prediction caching, and batch updates.

### Visual Legend
When debugging motion and tracking, the following color codes are typically used:
*   **Blue line**: Global Camera Motion trajectory.
*   **Pink arrows**: Kalman Filter velocity vector.
*   **Pink BBox**: Last Kalman Filter bounding box.
*   **Yellow BBox**: Last Detection bounding box.
*   **Red BBox**: Current Detection bounding box.
*   **Blue BBox**: Current Kalman Filter predicted bounding box.
*   **Green BBox**: Current Kalman Filter + Camera Motion compensated bounding box.

### TODO
*   Optimize GMC feature tracking for high-resolution 4K inputs.
*   Evaluate EKF (Extended Kalman Filter) if non-linear motion models are required.
