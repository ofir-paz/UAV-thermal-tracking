import csv
from typing import Dict, Tuple
from .overlays import Overlay, BoundingBox

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