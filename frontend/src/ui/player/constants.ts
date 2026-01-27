/**
 * @fileoverview Constants for player UI scaling and video cropping.
 */

/**
 * Minimum crop height fraction relative to the rotated source video height.
 * `scale` in TrackDetection is interpreted as this normalized crop height.
 */
export const MIN_CROP_NORM = 0.05

/**
 * Maximum crop height fraction relative to the rotated source video height.
 */
export const MAX_CROP_NORM = 1.0

/**
 * Baseline applied to detailed-mode "1.0" zoom so the default view/export is slightly zoomed out.
 */
export const DEFAULT_ZOOM_BASELINE = 0.9
