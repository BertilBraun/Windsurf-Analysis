/**
 * Utility for clamping numeric values.
 */

/**
 * Restricts a number to be within the specified inclusive range.
 *
 * @param value - The value to clamp.
 * @param min - The lower bound.
 * @param max - The upper bound.
 * @returns The clamped value.
 */
export function clamp(value: number, min: number, max: number) {
    return Math.max(min, Math.min(max, value))
}
