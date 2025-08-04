from typing import Callable
import cv2
import numpy as np

def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converts a frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def canny_edge_detector(threshold1: int = 100, threshold2: int = 200) -> Callable[[np.ndarray], np.ndarray]:
    def detector(frame: np.ndarray) -> np.ndarray:
        """Applies the Canny edge detector to a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        # Convert single-channel edges back to 3-channel BGR to allow color overlays
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return detector
