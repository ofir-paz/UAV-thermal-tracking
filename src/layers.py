from typing import Any, Dict, List, Literal, Tuple, Optional, Callable
from collections import deque
import cv2 as cv
import numpy as np
from video_player import Line, BoundingBox, OverlayItem 


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

    def get_stereo_warped_frame(self, frame: np.ndarray, state: Dict[Any, Any]) -> np.ndarray:    
        is_extracted = self._extract_state_metadata(frame, state)
        if not is_extracted:
            return frame
        
        h, w = frame.shape[:2]
    
        H_to_last, _ = cv.findHomography(self._current_pts, self._last_pts, cv.RANSAC, 3.0)
        self._H @= H_to_last
        warped = cv.warpPerspective(frame, self._H, (w, h))
        return warped

    def get_stereo_rectified_frame(self, frame: np.ndarray, state: Dict[Any, Any]) -> np.ndarray:
        is_extracted = self._extract_state_metadata(frame, state)
        if not is_extracted:
            return frame
        
        h, w = frame.shape[:2]
        
        # 1) FUNDAMENTAL (8-point via RANSAC)
        F, inliers = cv.findFundamentalMat(self._last_pts, self._current_pts, method=cv.FM_RANSAC,
                                        ransacReprojThreshold=1.0, confidence=0.999)
        inliers = inliers.ravel().astype(bool)
        pts1_i, pts2_i = self._last_pts[inliers], self._current_pts[inliers]
        assert F is not None and F.shape == (3,3), "F estimation failed."

        # 2) RECTIFY UNCALIBRATED (returns H1,H2)
        ok, H1, H2 = cv.stereoRectifyUncalibrated(pts1_i, pts2_i, F, imgSize=(w, h))
        assert ok, "Uncalibrated rectification failed."

        # Warp both images to the rectified domain
        rect1 = cv.warpPerspective(self._last_frame, H1, (w, h))
        rect2 = cv.warpPerspective(frame, H2, (w, h))

        # 3) DISPARITY (SGBM works well)
        # numDisparities must be divisible by 16; tune for your baseline/scene.
        minDisp = 0
        numDisp = 64  # try 64/96/128 depending on scene
        blk = 5
        sgbm = cv.StereoSGBM.create(minDisparity=minDisp,
                                    numDisparities=numDisp,
                                    blockSize=blk,
                                    P1=8*1*blk*blk,
                                    P2=32*1*blk*blk,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=50,
                                    speckleRange=1)
        disp = sgbm.compute(rect1, rect2).astype(np.float32) / 16.0  # disparity in pixels

        # 4) WARP rectified frame2 ONTO rectified frame1 using disparity
        # For rectified pairs: x_left ~ x_right + disp  =>  x_right = x_left - disp
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                            np.arange(w, dtype=np.float32), indexing='ij')
        map_x = (xx - disp).astype(np.float32)
        map_y = yy.astype(np.float32)
        aligned2_on_1 = cv.remap(rect2, map_x, map_y, interpolation=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT, borderValue=0)
        
        return aligned2_on_1


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

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        crop_h = int(h * self.crop_percentage)
        crop_w = int(w * self.crop_percentage)
        return frame[crop_h:h - crop_h, crop_w:w - crop_w]


