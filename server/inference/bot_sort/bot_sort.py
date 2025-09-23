import numpy as np
from collections import deque

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from server.inference.src.common_types import Detection


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        self.external_last_offset: np.ndarray | None = None
        # Debug visualization options
        self.debug_vis = getattr(args, 'debug_vis', True)
        self.debug_trail_len = int(getattr(args, 'debug_trail_len', 30))
        self.camera_translation_history = deque([], maxlen=self.debug_trail_len)

    def update(
        self,
        output_results: list[Detection],
        last_detections: list[Detection],
        external_warp: np.ndarray,
        debug_image: np.ndarray | None = None,
    ):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            scores = np.array([d.confidence for d in output_results])
            bboxes = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in output_results])

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []

        if len(dets) > 0:
            """Detections"""
            features = np.array([d.embedding for d in output_results])
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for (tlbr, s, f) in zip(dets, scores_keep, features)]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Snapshot KF-only predictions prior to applying external warp
        kf_tlwh_by_id = {track.track_id: track.tlwh.copy() for track in strack_pool}

        # Apply external camera motion compensation
        H = external_warp.astype(np.float64, copy=False)
        STrack.multi_gmc(strack_pool, H)
        STrack.multi_gmc(unconfirmed, H)

        # Debug visualization only if an image is provided
        if self.debug_vis and (debug_image is not None):
            import cv2

            # Update camera trail history (accumulate per-frame deltas)
            t = H[:2, 2]
            self.camera_translation_history.append(np.array([float(t[0]), float(t[1])], dtype=float))

            to_display = debug_image.copy()
            img_h, img_w = to_display.shape[:2]

            # Draw camera motion trail with COLORMAP_AUTUMN and highlight current point
            if len(self.camera_translation_history) > 0:
                center = (img_w // 2, img_h // 2)
                pts = [center]
                for dt in self.camera_translation_history:
                    last_pt = pts[-1]
                    nxt = (int(round(last_pt[0] + dt[0])), int(round(last_pt[1] + dt[1])))
                    pts.append(nxt)

                n = len(pts)
                # Build colormap colors from 0..255 mapped along the trail
                if n > 1:
                    idx = (np.linspace(0, 255, n - 1)).astype(np.uint8)
                    cmap = cv2.applyColorMap(idx, cv2.COLORMAP_AUTUMN)  # shape (n-1,1,3)
                    colors = [tuple(int(c) for c in cmap[i, 0, ::-1]) for i in range(n - 1)]  # BGR from RGB
                else:
                    colors = [(0, 255, 255)]

                for i in range(1, n):
                    cv2.line(to_display, pts[i - 1], pts[i], colors[i - 1], 2)
                # Mark most recent point
                cv2.circle(to_display, pts[-1], 3, (0, 0, 0), -1)
                cv2.circle(to_display, pts[-1], 2, (0, 255, 255), -1)

            def draw_box_with_label(image, tlwh, color, label):
                x, y, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                ty = max(y - 3, 0)
                cv2.putText(image, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            for track in strack_pool:
                bbox = track.tlwh  # externally-warped bbox
                draw_box_with_label(to_display, bbox, (0, 255, 0), f'KF + GMC id={track.track_id}')

                kf_bbox = kf_tlwh_by_id.get(track.track_id)
                if kf_bbox is not None:
                    draw_box_with_label(to_display, kf_bbox, (255, 0, 0), f'KF id={track.track_id}')

            for i, det in enumerate(detections):
                bbox = det.tlwh
                cv2.rectangle(
                    to_display,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    to_display,
                    f'Det {i}',
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            for out in output_results:
                bbox = out.bbox
                cv2.rectangle(
                    to_display,
                    (int(bbox.x1), int(bbox.y1)),
                    (int(bbox.x2), int(bbox.y2)),
                    (0, 0, 255),
                    2,
                )
            for out in last_detections:
                bbox = out.bbox
                cv2.rectangle(
                    to_display,
                    (int(bbox.x1), int(bbox.y1)),
                    (int(bbox.x2), int(bbox.y2)),
                    (0, 255, 255),
                    2,
                )

            # Build distance heatmaps between tracks (rows) and detections (cols)
            def build_heatmap(mat: np.ndarray, title: str, row_labels, col_labels) -> np.ndarray:
                if mat is None or mat.size == 0:
                    canvas = np.full((120, 240, 3), 30, dtype=np.uint8)
                    cv2.putText(
                        canvas,
                        f'{title}: no data',
                        (8, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    return canvas
                m = mat.astype(np.float32)
                m_min, m_max = float(np.min(m)), float(np.max(m))
                norm = (
                    np.zeros_like(m, dtype=np.uint8)
                    if (m_max - m_min) < 1e-6
                    else ((m - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
                )
                heat = cv2.applyColorMap(norm, cv2.COLORMAP_AUTUMN)
                cell_h, cell_w = 50, 50
                hm = cv2.resize(heat, (heat.shape[1] * cell_w, heat.shape[0] * cell_h), interpolation=cv2.INTER_AREA)
                top_margin, left_margin = 50, 36
                colorbar_w, colorbar_gap = 16, 8
                canvas_h = hm.shape[0] + top_margin
                canvas_w = hm.shape[1] + left_margin + colorbar_gap + colorbar_w
                canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)
                canvas[top_margin:, left_margin : left_margin + hm.shape[1]] = hm
                rows, cols = m.shape
                for r in range(rows + 1):
                    y = top_margin + r * cell_h
                    cv2.line(canvas, (left_margin, y), (left_margin + cols * cell_w, y), (60, 60, 60), 1)
                for c in range(cols + 1):
                    x = left_margin + c * cell_w
                    cv2.line(canvas, (x, top_margin), (x, top_margin + rows * cell_h), (60, 60, 60), 1)
                cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                for r, lbl in enumerate(row_labels):
                    y = top_margin + r * cell_h + int(cell_h * 0.7)
                    cv2.putText(
                        canvas,
                        f'Track {str(lbl)}',
                        (4, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                for c, lbl in enumerate(col_labels):
                    x = left_margin + c * cell_w + 2
                    cv2.putText(
                        canvas,
                        f'Det {str(lbl)}',
                        (x, top_margin - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                bar_h = hm.shape[0]
                grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
                grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_AUTUMN)
                x0 = left_margin + hm.shape[1] + colorbar_gap
                canvas[top_margin:, x0 : x0 + colorbar_w] = cv2.resize(
                    grad_color, (colorbar_w, bar_h), interpolation=cv2.INTER_AREA
                )
                cv2.putText(
                    canvas,
                    'high',
                    (x0 - 2, top_margin + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    'low',
                    (x0 + 2, top_margin + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f'{m_max:.2f}',
                    (x0 + colorbar_w + 4, top_margin + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f'{m_min:.2f}',
                    (x0 + colorbar_w + 4, top_margin + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                return canvas

            try:
                ious_debug = matching.iou_distance(strack_pool, detections)
                ious_debug_fused = matching.fuse_score(ious_debug.copy(), detections)
                emb_debug = matching.embedding_distance(strack_pool, detections) / 2.0
                ious_mask_debug = ious_debug > self.proximity_thresh
                emb_debug_clipped = emb_debug.copy()
                emb_debug_clipped[emb_debug_clipped > self.appearance_thresh] = 1.0
                emb_debug_clipped[ious_mask_debug] = 1.0
                final_debug = np.minimum(ious_debug_fused, emb_debug_clipped)

                row_labels = [t.track_id for t in strack_pool] if len(strack_pool) else []
                col_labels = list(range(len(detections))) if len(detections) else []

                hm_iou = build_heatmap(ious_debug, 'IoU distance', row_labels, col_labels)
                hm_emb = build_heatmap(emb_debug, 'Embedding distance', row_labels, col_labels)
                hm_final = build_heatmap(final_debug, 'Final distance', row_labels, col_labels)

                heat_row = np.concatenate([hm_iou, hm_emb, hm_final], axis=1)
                target_h = 140
                heat_row = cv2.resize(heat_row, (img_w, target_h), interpolation=cv2.INTER_AREA)
                cv2.imshow('dist_heatmaps (lower = better)', heat_row)
                try:
                    x, y, w, h = cv2.getWindowImageRect('strack_pool')
                    cv2.moveWindow('dist_heatmaps (lower = better)', x, y + h)
                except Exception:
                    pass
            except Exception:
                pass

            cv2.imshow('strack_pool', to_display)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = matching.fuse_score(ious_dists, detections)

        emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
        raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        # Popular ReID method (JDE / FairMOT)
        # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
        # emb_dists = dists

        # IoU making ReID
        # dists = matching.embedding_distance(strack_pool, detections)
        # dists[ious_dists_mask] = 1.0

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
        else:
            dets_second = []
            scores_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = matching.fuse_score(ious_dists, detections)

        emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
        raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
