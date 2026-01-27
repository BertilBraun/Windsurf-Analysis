/**
 * Utilities for extracting and indexing video packet metadata.
 */

import { EncodedPacketSink } from 'mediabunny'

/**
 * Metadata representing an encoded video packet.
 */
export type VideoPacketMeta = {
    /** Presentation timestamp (PTS). */
    ts: number
    /** Duration of the packet. */
    dur: number
    /** Whether the packet is a keyframe. */
    key: boolean
    /** Sequence number used to break ties for identical timestamps. */
    tie: number
}

/**
 * Extracts metadata from a video track and returns it sorted by timestamp and sequence number.
 *
 * @param videoTrack - The source video track to extract packets from.
 * @returns A promise resolving to an array of sorted packet metadata.
 */
export async function getSortedVideoPacketMeta(videoTrack: any): Promise<VideoPacketMeta[]> {
    try {
        const packetSink = new EncodedPacketSink(videoTrack)
        const packets: VideoPacketMeta[] = []
        for await (const pkt of packetSink.packets(undefined, undefined, { metadataOnly: true })) {
            if (pkt.timestamp < 0) continue
            packets.push({
                ts: pkt.timestamp,
                dur: pkt.duration ?? 0,
                key: pkt.type === 'key',
                tie: pkt.sequenceNumber ?? 0,
            })
        }
        packets.sort((a, b) => a.ts - b.ts || a.tie - b.tie)
        return packets
    } catch {
        return []
    }
}

/**
 * Performs a binary search to find the index of the timestamp closest to the target time.
 *
 * @param ptsSec - Sorted array of presentation timestamps in seconds.
 * @param tSec - Target time in seconds.
 * @returns The index of the timestamp closest to the target time.
 */
export function closestIndexForTimestampSec(ptsSec: number[], tSec: number): number {
    const n = ptsSec.length
    if (n <= 0) return 0
    if (tSec <= ptsSec[0]!) return 0
    if (tSec >= ptsSec[n - 1]!) return n - 1

    // Lower-bound: first i where pts[i] >= tSec
    let lo = 0
    let hi = n - 1
    while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (ptsSec[mid]! < tSec) lo = mid + 1
        else hi = mid
    }

    const i1 = lo
    const i0 = Math.max(0, i1 - 1)
    const d0 = Math.abs(tSec - ptsSec[i0]!)
    const d1 = Math.abs(ptsSec[i1]! - tSec)
    return d0 <= d1 ? i0 : i1
}
