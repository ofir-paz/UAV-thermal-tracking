from typing import Any, Dict, List, Literal, Tuple, Optional, Callable
from collections import deque
import cv2 as cv
import numpy as np
from config import Classes
from video_player import Line, BoundingBox, OverlayItem, Color, np_to_overlay_items
from sort import Sort


class MedianFilter:
    """
    Median filter to remove noise from a gray scale image.
    """
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        # Apply median blur to remove noise
        filtered = cv.medianBlur(frame, self.kernel_size)
        return filtered


class HighPassFilter:
    """
    High-pass filter to remove low-frequency noise from a gray scale image.
    """
    def __init__(self, cutoff: float = 3):
        self.cutoff = cutoff

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        # Apply Gaussian blur to remove low frequencies
        blurred = cv.GaussianBlur(frame, (0, 0), self.cutoff)
        # Subtract blurred image from original to get high frequencies
        high_passed = cv.subtract(frame, blurred)
        return high_passed


class BandPassFilter:
    """
    Band-pass filter to remove both low and high-frequency noise from a gray scale image.
    """
    def __init__(self, low_cutoff: float = 0.2, high_cutoff: float = 3):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        
        low_passed = cv.GaussianBlur(frame, (0, 0), self.low_cutoff)
        high_passed = HighPassFilter(self.high_cutoff)(low_passed)
        return high_passed


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

    def __call__(self, gray_frame: np.ndarray, state: Dict[Any, Any]) -> List[OverlayItem]:
        assert gray_frame.ndim == 2, "Input frame must be grayscale (2D array)."
        
        # If we don't have points yet, seed them
        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) == 0:
            self._seed_points(gray_frame)
            state["current_pts"] = self._prev_pts.copy().reshape(-1, 2).astype(np.float32)
            return self._tracks_to_lines()

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self._prev_gray, gray_frame, self._prev_pts, None, **self.lk_params)
        alive_mask = st.reshape(-1).astype(bool)

        items: List[OverlayItem] = []
        if p1 is not None and st is not None and np.any(st == 1):
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

        # Add found tracked points to state
        state["last_alive"] = alive_mask
        state["current_pts"] = self._prev_pts.copy().reshape(-1, 2).astype(np.float32)
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


class MotionStabilizer:
    def __init__(self) -> None:
        self._first_frame: np.ndarray
        self._last_frame: np.ndarray
        self._current_frame: np.ndarray

        self._first_pts: np.ndarray
        self._last_pts: np.ndarray
        self._current_pts: np.ndarray
        
        self._H: np.ndarray = np.eye(3, dtype=np.float32)  # Homography matrix

    def _extract_state_metadata(self, frame: np.ndarray, state: Dict[Any, Any]) -> bool:
        """Extracts and updates state metadata from the current frame."""
        if self.__dict__.get("_first_frame") is None:  # First call, no previous frame
            self._first_frame = self._last_frame = self._current_frame = frame.copy()
            self._first_pts = self._last_pts = self._current_pts = state["current_pts"]
            return False
        
        self._last_frame = self._current_frame
        self._last_pts = self._current_pts[state["last_alive"]]
        self._current_frame = frame
        self._current_pts = state["current_pts"]
        return True

    def warp_back(self, frame: np.ndarray) -> np.ndarray:
        """Warps the frame back to the original size."""
        if self.__dict__.get("_H") is None:
            return frame

        h, w = frame.shape[:2]
        warped_to_first = cv.warpPerspective(frame, np.linalg.inv(self._H), (w, h))
        return warped_to_first

    def get_stereo_warped_frame(self, frame: np.ndarray, state: Dict[Any, Any]) -> Tuple[np.ndarray, Optional[Callable]]:    
        is_extracted = self._extract_state_metadata(frame, state)
        if not is_extracted:
            return frame, None
        
        h, w = frame.shape[:2]
    
        H_to_last, _ = cv.findHomography(self._current_pts, self._last_pts, cv.RANSAC, 3.0)
        if H_to_last is None:
            return frame, None

        self._H @= H_to_last
        warped = cv.warpPerspective(frame, self._H, (w, h))

        def warp_func(coords: np.ndarray) -> np.ndarray:
            H_inv = np.linalg.inv(self._H)
            warped_coords = cv.perspectiveTransform(np.array([coords]), H_inv)
            return warped_coords[0]

        return warped, warp_func


class BackgroundSubtraction:
    def __init__(self, method: Literal["KNN", "MOG"] = "KNN") -> None:
        if method == "KNN":
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(history=200, dist2Threshold=500, detectShadows=False)
        elif method == "MOG":
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask


def get_morphological_op(open_size: int = 3, close_size: int = 4) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a morphological operation function with the specified kernel size."""
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (round(open_size * 0.67), round(open_size * 1.33)))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (round(close_size * 0.67), round(close_size * 1.33)))
    last_dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size + 1, close_size + 1))

    def morphological_op(frame: np.ndarray) -> np.ndarray:
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel_open)
        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel_close)
        frame = cv.morphologyEx(frame, cv.MORPH_ERODE, kernel_open)
        frame = cv.morphologyEx(frame, cv.MORPH_DILATE, last_dilate_kernel)
        return frame
    
    return morphological_op


class CropImage:
    """
    Crops the image to a specified percentage of its original size.
    """
    def __init__(self, crop_percentage: float = 0.1):
        self.crop_percentage = crop_percentage
        self.crop_h = 0
        self.crop_w = 0

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        h, w = frame.shape[:2]
        self.crop_h = int(h * self.crop_percentage)
        self.crop_w = int(w * self.crop_percentage)
        
        cropped_frame = frame[self.crop_h:h - self.crop_h, self.crop_w:w - self.crop_w]
        
        def warp_func(coords: np.ndarray) -> np.ndarray:
            return coords + np.array([self.crop_w, self.crop_h])
            
        return cropped_frame, warp_func


class DetectClasses:
    def __init__(
        self, 
        dilate_size: int = 8, 
        max_size: int = 50, 
        max_ratio: float = 2.0,
        max_age: int = 5,
        min_hits: int = 20,
        iou_threshold: float = 0.3,
        init_frame_num: int = 10
    ) -> None:
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_size, dilate_size))
        self.max_size = max_size
        self.max_ratio = max_ratio
        self.frame_num = 0
        self.init_frame_num = init_frame_num

    def __call__(self, frame: np.ndarray) -> List[OverlayItem]:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        
        # Skip processing for the first few frames
        self.frame_num += 1
        if self.frame_num < self.init_frame_num:
            return []
        
        frame = cv.morphologyEx(frame, cv.MORPH_DILATE, self.dilate_kernel)
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Fill bounding boxes for each contour
        bounding_boxes: List[OverlayItem] = []
        dets = np.empty((0, 5), dtype=np.int32)  # (x1, y1, x2, y2, score)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            klass = self._classify_contour(x, y, w, h)
            if klass == Classes.BACKGROUND:
                continue
            bounding_boxes.append(BoundingBox(x, y, w, h, label=klass.TEXT, color=Color(klass.COLOR)))
            dets = np.append(dets, [[x, y, x + w, y + h, 0]], axis=0)  # Score is unused in sort

        tracked_objects = self.sort_tracker.update(dets)
        if tracked_objects.shape[0] > 0:
            tracked_objects[:, 2] -= tracked_objects[:, 0]
            tracked_objects[:, 3] -= tracked_objects[:, 1]
        return np_to_overlay_items(tracked_objects, BoundingBox)

    def _classify_contour(self, x: int, y: int, w: int, h: int) -> Classes:
        """Classifies the contour based on its position and size."""
        if self._filter_contour(x, y, w, h):
            return Classes.BACKGROUND
        elif w * h > 20 * 20:
            return Classes.VEHICLE
        elif h / w > 0.95:
            return Classes.PERSON
        else:
            return Classes.BACKGROUND

    def _filter_contour(self, x: int, y: int, w: int, h: int) -> bool:
        """Filters out contours that are too small or too large."""
        return (
            w > self.max_size or 
            h > self.max_size or
            w / h > self.max_ratio or
            h / w > self.max_ratio
        )
