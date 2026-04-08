"""
MIT-compatible lightweight SORT-style tracker for this repository.

This implementation keeps the same external API used by TrackDetectedObjects:
    Sort(max_age, min_hits, iou_threshold).update(dets)
where dets is an array of shape (N, 5) [x1,y1,x2,y2,score] and return value
is shape (M, 5) [x1,y1,x2,y2,track_id].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray  # [x1,y1,x2,y2]
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        self.bbox = bbox.astype(np.float32, copy=True)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: List[_Track] = []
        self.frame_count = 0
        self._next_id = 1

    def _match(self, detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(self.trackers) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.trackers)))
        matches: List[Tuple[int, int]] = []

        # Greedy IoU matching by highest IoU first.
        pairs: List[Tuple[float, int, int]] = []
        for d_idx, det in enumerate(detections):
            det_box = det[:4]
            for t_idx, trk in enumerate(self.trackers):
                pairs.append((_iou(det_box, trk.bbox), d_idx, t_idx))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou, d_idx, t_idx in pairs:
            if iou < self.iou_threshold:
                break
            if d_idx in unmatched_dets and t_idx in unmatched_tracks:
                matches.append((d_idx, t_idx))
                unmatched_dets.remove(d_idx)
                unmatched_tracks.remove(t_idx)

        return matches, sorted(unmatched_dets), sorted(unmatched_tracks)

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        self.frame_count += 1

        if dets is None:
            dets = np.empty((0, 5), dtype=np.float32)
        dets = np.asarray(dets, dtype=np.float32)
        if dets.size == 0:
            dets = np.empty((0, 5), dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        for trk in self.trackers:
            trk.mark_missed()

        matches, unmatched_dets, _ = self._match(dets)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(dets[d_idx, :4])

        for d_idx in unmatched_dets:
            self.trackers.append(
                _Track(
                    track_id=self._next_id,
                    bbox=dets[d_idx, :4].astype(np.float32, copy=True),
                )
            )
            self._next_id += 1

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        outputs = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(np.array([*trk.bbox.tolist(), float(trk.track_id)], dtype=np.float32))

        if not outputs:
            return np.empty((0, 5), dtype=np.float32)
        return np.vstack(outputs)
