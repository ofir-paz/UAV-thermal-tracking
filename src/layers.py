from typing import Any, Dict, List, Literal, Tuple, Optional, Callable, Union, Deque
from collections import deque, defaultdict
from enum import Enum
import cv2 as cv
import numpy as np
import supervision as sv
from config import Classes, VideosConfig
from video_player import Line, BoundingBox, OverlayItem, Color, Text, np_to_overlay_items


class MedianFilter:
    """
    Drop-in optimized median filter.

    Behavior:
      - If num_history_frames <= 1:
          * kernel_size > 1  -> spatial median via cv.medianBlur
          * kernel_size <= 1 -> returns the frame unchanged
      - Else:
          * temporal median across 'num_history_frames' frames sampled every
            'history_dilation' calls (no spatial median in this branch,
            same as original code).
    """
    def __init__(self, kernel_size: int = 1, num_history_frames: int = 5, history_dilation: int = 2):
        assert num_history_frames >= 1, "num_history_frames must be >= 1"
        assert history_dilation >= 1, "history_dilation must be >= 1"

        # Ensure odd kernel (OpenCV requires positive odd for medianBlur)
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1:
            self.kernel_size = 1
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.num_history_frames = int(num_history_frames)
        self.history_dilation   = int(history_dilation)

        # Ring buffer for sampled frames (shape will be allocated on first call)
        self._ring = None          # np.ndarray of shape (N, H, W), dtype matches input
        self._ring_write_idx = 0   # next write position in ring
        self._ring_filled = 0      # number of valid frames currently in ring
        self._call_count = 0       # to apply dilation
        self._cached_temporal = None
        self._last_update_was_sample = False

    def _ensure_ring(self, frame: np.ndarray) -> None:
        """Allocate/resize ring buffer to match frame shape/dtype."""
        if (self._ring is None or
            self._ring.shape[1:] != frame.shape or
            self._ring.dtype != frame.dtype):
            self._ring = np.empty((self.num_history_frames, *frame.shape), dtype=frame.dtype)
            self._ring_write_idx = 0
            self._ring_filled = 0

    def _temporal_median(self, sample: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel median across axis=0 of 'sample' with shape (T,H,W)
        using np.partition (O(T) per pixel). Assumes T == self.num_history_frames.
        """
        T = sample.shape[0]
        # For odd T we need the kth element; for even we average the two middles.
        k = T // 2
        if T % 2 == 1:
            # Partition at k; the kth slice holds the median values at each pixel.
            part = np.partition(sample, kth=k, axis=0)
            return part[k]
        else:
            # Partition at both k-1 and k to get the two middle order statistics.
            part = np.partition(sample, kth=(k-1, k), axis=0)
            lo = part[k-1]
            hi = part[k]
            # Average safely (avoid uint8 overflow):
            if sample.dtype == np.uint8:
                return ((lo.astype(np.uint16) + hi.astype(np.uint16)) // 2).astype(np.uint8)
            else:
                return ((lo.astype(np.float32) + hi.astype(np.float32)) * 0.5).astype(sample.dtype, copy=False)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        # Make sure strides are sane for fast ops
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        # Fast paths when not using temporal history
        if self.num_history_frames <= 1:
            if self.kernel_size > 1:
                return cv.medianBlur(frame, self.kernel_size)
            return frame  # kernel_size == 1 ➜ identity

        # Temporal mode
        self._ensure_ring(frame)
        self._call_count += 1

        # Sample only every 'history_dilation' frames
        updated = False
        if (self._call_count % self.history_dilation) == 0:
            self._ring[self._ring_write_idx] = frame
            self._ring_write_idx = (self._ring_write_idx + 1) % self.num_history_frames
            if self._ring_filled < self.num_history_frames:
                self._ring_filled += 1
            updated = True

        # Warm-up
        if self._ring_filled < self.num_history_frames:
            return frame

        # Recompute only when we actually added a new sample
        if updated or self._cached_temporal is None:
            T = self._ring.shape[0]
            if T == 3:
                A, B, C = self._ring[0], self._ring[1], self._ring[2]
                lo = np.minimum(np.minimum(A, B), C)
                hi = np.maximum(np.maximum(A, B), C)
                S  = A.astype(np.uint16) + B.astype(np.uint16) + C.astype(np.uint16)
                self._cached_temporal = (S - lo.astype(np.uint16) - hi.astype(np.uint16)).astype(np.uint8)
            else:
                # fallback (odd/even T) using partition as you had
                self._cached_temporal = self._temporal_median(self._ring)

        return self._cached_temporal

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
        return_overlay_items: bool = True
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
        self.return_overlay_items = return_overlay_items

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
        items = self._tracks_to_lines() if self.return_overlay_items else []

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
    def __init__(self, crop_percentage: Optional[float] = 0.1, fixer_ema_factor: float = 1) -> None:
        self._first_frame: np.ndarray
        self._last_frame: np.ndarray
        self._current_frame: np.ndarray

        self._first_pts: np.ndarray
        self._last_pts: np.ndarray
        self._current_pts: np.ndarray
        
        self._H: np.ndarray = np.eye(3, dtype=np.float32)  # Homography matrix
        self.crop_percentage = crop_percentage
        self.fixer_ema_factor = fixer_ema_factor
        self._homography_fix = np.eye(2, 3, dtype=np.float32)

    def _extract_state_metadata(self, frame: np.ndarray, state: Dict[Any, Any]) -> bool:
        """Extracts and updates state metadata from the current frame."""
        if self.__dict__.get("_first_frame") is None:  # First call, no previous frame
            self._first_frame = self._last_frame = self._current_frame = frame.copy()
            self._first_pts = self._last_pts = self._current_pts = state["current_pts"]
            return False

        self._first_pts = self._first_pts[state["last_alive"]]
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

    def get_corrected_frame(self, frame: np.ndarray, state: Dict[Any, Any]) -> Tuple[np.ndarray, Optional[Callable]]:    
        is_extracted = self._extract_state_metadata(frame, state)
        if not is_extracted:
            return frame, None
        
        h, w = frame.shape[:2]
    
        H_ransac, inlier_mask = cv.findHomography(self._current_pts, self._last_pts, cv.RANSAC, 3.0)
        inliers = inlier_mask.ravel().astype(bool)
        H_to_last, _ = cv.findHomography(self._current_pts[inliers], self._last_pts[inliers], 0, 3.0)

        self._H @= H_to_last
        self._H /= (self._H[2, 2] + 1e-12)  # Normalize
        
        warped = cv.warpPerspective(frame, self._H, (w, h))

        if self.fixer_ema_factor < 1:
            _a = self.fixer_ema_factor
            warped_points = cv.perspectiveTransform(self._current_pts[inliers].reshape(-1, 1, 2), self._H)
            affine_2, _ = cv.estimateAffine2D(warped_points, self._first_pts[inliers], method=cv.LMEDS, refineIters=100)
            self._homography_fix = _a * self._homography_fix + (1 - _a) * affine_2
            warped = cv.warpAffine(warped, self._homography_fix, (w, h))

        def warp_func(coords: np.ndarray) -> np.ndarray:
            H_inv = np.linalg.inv(self._H)
            warped_coords = cv.perspectiveTransform(np.array([coords]), H_inv)
            return warped_coords[0]

        return warped, warp_func
    
    def post_warp_crop(self, frame: np.ndarray) -> np.ndarray:
        """Crops the warped frame to noisy borders."""
        if not self.crop_percentage:
            return frame

        h, w = frame.shape[:2]
        mask = np.zeros_like(frame, dtype=np.uint8)
        crop_h = int(h * self.crop_percentage)
        crop_w = int(w * self.crop_percentage)
        mask[crop_h:h - crop_h, crop_w:w - crop_w] = 255
        mask = cv.warpPerspective(mask, self._H, (w, h))
        cropped_frame = cv.bitwise_and(frame, mask)
        return cropped_frame

class BackgroundSubtraction:
    def __init__(self, method: Literal["KNN", "MOG"] = "KNN") -> None:
        if method == "KNN":
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=600, detectShadows=False)
        elif method == "MOG":
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(history=150, varThreshold=25, detectShadows=False)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask


def get_morphological_op(open_size: int = 3, close_size: int = 4, last_dilation_kernel: Tuple[int, int] = (2, 3)) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a morphological operation function with the specified kernel size."""
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (round(open_size * 0.67), round(open_size * 1.33)))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (round(close_size * 0.67), round(close_size * 1.33)))
    last_dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, last_dilation_kernel)

    def morphological_op(frame: np.ndarray) -> np.ndarray:
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel_open)
        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel_close)
        frame = cv.morphologyEx(frame, cv.MORPH_DILATE, last_dilate_kernel)
        return frame
    
    return morphological_op


class CropImage:
    """
    Crops the image to a specified percentage of its original size.
    """
    def __init__(self, crop_percentage: float = 0.1, crop_with_mask: bool = False):
        self.crop_percentage = crop_percentage
        self.crop_h = 0
        self.crop_w = 0
        self.crop_with_mask = crop_with_mask

    def __call__(self, frame: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]]:
        h, w = frame.shape[:2]
        self.crop_h = int(h * self.crop_percentage)
        self.crop_w = int(w * self.crop_percentage)
        
        if self.crop_with_mask:
            mask = np.zeros_like(frame, dtype=np.uint8)
            mask[self.crop_h:h - self.crop_h, self.crop_w:w - self.crop_w] = 255
            frame = cv.bitwise_and(frame, mask)
            return frame

        else:
            cropped_frame = frame[self.crop_h:h - self.crop_h, self.crop_w:w - self.crop_w]
            
            def warp_func(coords: np.ndarray) -> np.ndarray:
                return coords + np.array([self.crop_w, self.crop_h])
            
            return cropped_frame, warp_func


class DetectClasses:
    def __init__(
        self, 
        dilate_size: int = 8, 
        max_size: int = 40, 
        max_ratio: float = 2.25,
        vehicle_min_size: int = 18 * 18,
        person_h_w_ratio: float = 0.925,
        init_frame_num: int = 10,
        return_overlay_items: bool = True
    ) -> None:
        self.dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate_size, dilate_size)) if dilate_size > 0 else None
        self.max_size = max_size
        self.max_ratio = max_ratio
        self.vehicle_min_size = vehicle_min_size
        self.person_h_w_ratio = person_h_w_ratio
        self.init_frame_num = init_frame_num
        self.return_overlay_items = return_overlay_items
        self.frame_num = 0

    def __call__(self, frame: np.ndarray, state: Dict[Any, Any]) -> List[OverlayItem]:
        assert frame.ndim == 2, "Input frame must be grayscale (2D array)."
        
        # Skip processing for the first few frames
        self.frame_num += 1
        if self.frame_num < self.init_frame_num:
            return []
        
        if self.dilate_kernel is not None:
            frame = cv.morphologyEx(frame, cv.MORPH_DILATE, self.dilate_kernel)
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Fill bounding boxes for each contour
        bounding_boxes: List[OverlayItem] = []
        xyxy = []
        class_id = []
        confidences = []

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            klass, score = self._classify_contour(x, y, w, h)
            if klass == Classes.BACKGROUND:
                continue
            
            if self.return_overlay_items:
                label = f"{score:.2f}"
                bounding_boxes.append(BoundingBox(x, y, w, h, label=label, color=klass.COLOR))
            
            xyxy.append([x, y, x + w, y + h])
            class_id.append(klass.value)
            confidences.append(score)

        state["detections"] = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            class_id=np.array(class_id, dtype=np.int32),
            confidence=np.array(confidences, dtype=np.float32)
        ) if xyxy else sv.Detections.empty()

        return bounding_boxes

    def _classify_contour(self, x: int, y: int, w: int, h: int) -> Tuple[Classes, float]:
        """Classifies the contour based on its position and size."""
        if self._filter_contour(x, y, w, h):
            return Classes.BACKGROUND, 0.0
        
        elif w * h > self.vehicle_min_size:
            size_score = min(w * h / (self.vehicle_min_size * 1.5), 1.0)
            ratio_score = w / (h + w)
            vehicle_score = (0.8 * size_score + 0.2 * ratio_score) / (size_score + ratio_score)
            return Classes.VEHICLE, vehicle_score
        
        elif h / w > self.person_h_w_ratio:
            size_score = -0.04 * np.sqrt(w * h) + 1.0
            ratio_score = min(h / (w * 1.4), 1.0)
            person_score = (0.6 * size_score + 0.4 * ratio_score) / (size_score + ratio_score)
            return Classes.PERSON, person_score
        
        else:
            return Classes.BACKGROUND, 0.0

    def _filter_contour(self, x: int, y: int, w: int, h: int) -> bool:
        """Filters out contours that are too small or too large."""
        return (
            w > self.max_size or 
            h > self.max_size or
            w / h > self.max_ratio or
            h / w > self.max_ratio
        )


class TrackDetectedObjects:
    """
    Tracks detected objects across frames using SORT.
    """
    class Library(Enum):
        SORT = "sort"
        TRACKERS = "trackers"

    class ObjectType:
        def __init__(self):
            self._counts = defaultdict(int)

        def was_detected_as(self, class_id: int) -> None:
            self._counts[class_id] += 1

        def get_class(self) -> int:
            if not self._counts:
                return Classes.BACKGROUND.value
            most_common_class = max(self._counts.items(), key=lambda item: item[1])[0]
            return most_common_class

    def __init__(
        self, 
        max_age: int = 5, 
        min_hits: int = 20, 
        iou_threshold: float = 0.3,
        score_threshold: float = 0.5,
        init_frame_num: int = 10,
        library: Literal["SORT", "Trackers"] = "SORT",
        **kwargs
    ) -> None:
        self._library = self.Library(library.lower())

        if self._library == self.Library.TRACKERS:
            from trackers import SORTTracker
            self.tracker = SORTTracker(
                lost_track_buffer=max_age, 
                frame_rate=kwargs.get("frame_rate", VideosConfig.FRAME_RATE),
                track_activation_threshold=score_threshold,
                minimum_consecutive_frames=min_hits,
                minimum_iou_threshold=iou_threshold
            )
        elif self._library == self.Library.SORT:
            from sort import Sort
            self.tracker = Sort(
                max_age=max_age, 
                min_hits=min_hits, 
                iou_threshold=iou_threshold,
            )
        else:
            raise ValueError(f"Unsupported library: {library}. Use 'SORT' or 'Trackers'.")

        self.object_types: Dict[int, self.ObjectType] = defaultdict(self.ObjectType)
        self.init_frame_num = init_frame_num
        self.frame_num = 0

    def __call__(self, frame: np.ndarray, state: Dict[Any, Any]) -> List[OverlayItem]:
        """
        Detections should be in the format (x1, y1, x2, y2, score).
        Returns a list of OverlayItems representing tracked objects.
        """
        # Skip processing for the first few frames
        self.frame_num += 1
        if self.frame_num < self.init_frame_num:
            return []
        
        _detections: sv.Detections = state.get("detections", sv.Detections.empty())

        if self._library == self.Library.TRACKERS:
            tracked_objects = self.tracker.update(_detections)

            bbs: List[OverlayItem] = []
            for (xyxy, _, confidence, class_id, tracker_id, _) in tracked_objects:
                tracker_id = int(tracker_id)
                if tracker_id == -1:
                    continue

                # Fix detected class ID
                self.object_types[tracker_id].was_detected_as(class_id)
                class_id = self.object_types[tracker_id].get_class()

                x = xyxy[0]; y = xyxy[1]
                w = xyxy[2] - x; h = xyxy[3] - y
                label = f"ID: {tracker_id} {Classes(class_id).TEXT} {confidence:.2f}"
                color = Classes(class_id).COLOR

                bbs.append(BoundingBox(x, y, w, h, label, color))
            
            return bbs

        elif self._library == self.Library.SORT:
            # Add dummy scores. SORT requires a score column, but it doesn't use it.
            dets = np.hstack((_detections.xyxy, np.zeros((_detections.xyxy.shape[0], 1))), dtype=np.float32)
            tracked_objects = self.tracker.update(dets)

            if tracked_objects.shape[0] > 0:
                tracked_objects[:, 2] -= tracked_objects[:, 0]
                tracked_objects[:, 3] -= tracked_objects[:, 1]

            return np_to_overlay_items(tracked_objects, BoundingBox)
    