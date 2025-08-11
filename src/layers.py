from typing import List, Tuple, Optional
from collections import deque
import cv2 as cv
import numpy as np
from video_player import Line, OverlayItem


class OpticalFlowLambda:
    """
    Callable that maintains LK optical-flow state across frames and returns overlay items.

    Returns per-call:
      - A list of Line items for motion segments (old->new)
      - Optionally current Point items (if draw_points=True)
      - Optionally short trails by keeping last N segments (trail_len > 0)

    Usage:
        flow_overlay = OpticalFlowLambda()
        video.add_online_overlay(name="KLT tracks", overlay_func=flow_overlay)
    """
    def __init__(
        self,
        feature_params: Optional[dict] = None,
        lk_params: Optional[dict] = None,
        max_corners: int = 100,
        reseed_min_points: int = 15,
        track_len: int = 60,              # max number of saved points per track
        avoid_reseed_radius: int = 10,     # radius (px) around live points to avoid when reseeding
        color_seed: int = 42,
    ):
        self.feature_params = feature_params or dict(
            maxCorners=max_corners,
            qualityLevel=0.25,
            minDistance=10,
            blockSize=9
        )
        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        self.max_corners = max_corners
        self.reseed_min_points = reseed_min_points
        self.track_len = int(track_len) if track_len and track_len > 1 else 2
        self.avoid_reseed_radius = int(avoid_reseed_radius)

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None   # shape (N,1,2), float32
        self._tracks: List[deque] = []               # list of deque([(x,y), ...], maxlen=track_len)
        self._colors: List[Tuple[int, int, int]] = []  # aligned with _tracks
        self._rng = np.random.default_rng(color_seed)

    def __call__(self, gray_frame: np.ndarray) -> List[OverlayItem]:
        assert gray_frame.ndim == 2, "Input frame must be grayscale (2D array)."
        
        # If we don't have points yet, seed them
        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) == 0:
            self._seed_points(gray_frame)
            return self._tracks_to_lines()

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self._prev_gray, gray_frame, self._prev_pts, None, **self.lk_params)

        items: List[OverlayItem] = []
        if p1 is not None and st is not None and np.any(st == 1):
            alive_mask = st.reshape(-1).astype(bool)
            new_pts = p1[alive_mask].reshape(-1, 2)          # (M,2)
            # Keep tracks/colors for alive points only, preserving order
            self._tracks = [t for t, alive in zip(self._tracks, alive_mask) if alive]
            self._colors = [c for c, alive in zip(self._colors, alive_mask) if alive]

            # Append new positions to each surviving track
            for t, (x, y) in zip(self._tracks, new_pts):
                t.append((round(x), round(y)))

            self._prev_gray = gray_frame.copy()
            self._prev_pts = new_pts.reshape(-1, 1, 2).astype(np.float32)

            # Reseed if we lost too many points
            if len(self._prev_pts) < self.reseed_min_points:
                self._reseed_more(gray_frame)

        else:
            # Couldn’t compute flow → reseed from scratch
            self._seed_points(gray_frame)

        # Convert all tracks with length>=2 to Line polylines
        items = self._tracks_to_lines()
        return items


    # ---------------- internal helpers ------------------
    def _seed_points(self, gray: np.ndarray):
        p0 = cv.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self._prev_gray = gray
        if p0 is None:
            self._prev_pts = np.empty((0, 1, 2), dtype=np.float32)
            self._tracks = []
            self._colors = []
            return

        self._prev_pts = p0.astype(np.float32)
        pts = self._prev_pts.reshape(-1, 2)
        self._tracks = [deque([(round(x), round(y))], maxlen=self.track_len) for (x, y) in pts]
        self._colors = [tuple(round(v) for v in self._rng.integers(0, 255, size=3)) for _ in self._tracks]

    def _reseed_more(self, gray: np.ndarray):
        """Top-up features to reach max_corners, avoiding current points’ neighborhoods."""
        need = self.max_corners - len(self._prev_pts)
        if need <= 0:
            return

        # Build a mask that zeros out disks around current points
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for t in self._tracks:
            x, y = t[-1]
            cv.circle(mask, (round(x), round(y)), self.avoid_reseed_radius, 0, -1)

        p_add = cv.goodFeaturesToTrack(gray, mask=mask, **{**self.feature_params, "maxCorners": need})
        if p_add is None:
            return

        p_add = p_add.astype(np.float32)
        # Concatenate to prev_pts
        if len(self._prev_pts) == 0:
            self._prev_pts = p_add
        else:
            self._prev_pts = np.vstack([self._prev_pts, p_add])

        # Create tracks/colors for the new points
        for (x, y) in p_add.reshape(-1, 2):
            self._tracks.append(deque([(round(x), round(y))], maxlen=self.track_len))
            self._colors.append(tuple(round(v) for v in self._rng.integers(0, 255, size=3)))

    def _tracks_to_lines(self) -> List[OverlayItem]:
        items: List[OverlayItem] = []
        for pts, color in zip(self._tracks, self._colors):
            if len(pts) >= 2:
                items.append(Line(points=list(pts), color=color, thickness=2))
        return items