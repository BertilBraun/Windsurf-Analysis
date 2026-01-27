/**
 * Math utilities for calculating layout and rendering dimensions for the player.
 */

/**
 * Represents a rectangle's position and dimensions relative to a container,
 * including the scale factor applied to the source.
 */
export type BaseRect = { x: number; y: number; w: number; h: number; scale: number }

/**
 * Computes the centered bounding box for a source (e.g., video) within a container
 * using a "contain" fit strategy to maintain aspect ratio.
 *
 * @param outW - The width of the output container.
 * @param outH - The height of the output container.
 * @param vidW - The intrinsic width of the source content.
 * @param vidH - The intrinsic height of the source content.
 * @returns The calculated position, dimensions, and scale factor.
 */
export function computeBaseRect(outW: number, outH: number, vidW: number, vidH: number): BaseRect {
    const scale = Math.min(outW / vidW, outH / vidH)
    const dispW = vidW * scale
    const dispH = vidH * scale
    const offX = (outW - dispW) / 2
    const offY = (outH - dispH) / 2
    return { x: offX, y: offY, w: dispW, h: dispH, scale }
}
