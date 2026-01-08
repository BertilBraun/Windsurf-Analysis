import { EncodedPacketSink } from 'mediabunny'

export type VideoPacketMeta = {
    ts: number
    dur: number
    key: boolean
    tie: number
}

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

