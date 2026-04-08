import csv
from typing import Dict, Tuple, Callable
import cv2
import numpy as np
from .overlays import Overlay, BoundingBox


def canny_edge_detector(threshold1: int = 100, threshold2: int = 200) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a transform function that applies Canny edge detection."""
    def _canny(frame: np.ndarray) -> np.ndarray:
        gray = frame
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, threshold1, threshold2)
    return _canny

def load_bounding_boxes_from_csv(filepath: str, overlay_name: str = "default_overlay") -> Dict[int, Dict[str, Overlay]]:
    """
    Loads bounding boxes from a CSV file.

    The CSV file should have the following columns:
    frame_number, x, y, width, height, label, color

    Args:
        filepath: The path to the CSV file.
        overlay_name: The name to assign to this set of overlays.

    Returns:
        A dictionary mapping frame numbers to a dictionary of Overlay objects (keyed by overlay_name).
    """
    overlays_by_frame: Dict[int, Dict[str, Overlay]] = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_number = int(row["frame_number"])
            x = int(row["x"])
            y = int(row["y"])
            width = int(row["width"])
            height = int(row["height"])
            label = row.get("label")
            color_str = row.get("color")
            color: Tuple[int, int, int] = tuple(map(int, color_str.split(','))) if color_str else (0, 255, 0)  # type: ignore

            bbox = BoundingBox(x, y, width, height, label, color)

            if frame_number not in overlays_by_frame:
                overlays_by_frame[frame_number] = {}
            
            if overlay_name not in overlays_by_frame[frame_number]:
                overlays_by_frame[frame_number][overlay_name] = Overlay(name=overlay_name, overlay_items=[])

            overlays_by_frame[frame_number][overlay_name].overlay_items.append(bbox)

    return overlays_by_frame
