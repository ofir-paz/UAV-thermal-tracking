
import csv
from typing import Dict
from .overlays import Overlay, BoundingBox

def load_bounding_boxes_from_csv(filepath: str) -> Dict[int, Overlay]:
    """
    Loads bounding boxes from a CSV file.

    The CSV file should have the following columns:
    frame_number, x, y, width, height, label, color

    Args:
        filepath: The path to the CSV file.

    Returns:
        A dictionary mapping frame numbers to Overlay objects.
    """
    overlays: Dict[int, Overlay] = {}
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
            color = tuple(map(int, color_str.split(','))) if color_str else (0, 255, 0)

            bbox = BoundingBox(x, y, width, height, label, color)

            if frame_number not in overlays:
                overlays[frame_number] = Overlay(bounding_boxes=[])
            overlays[frame_number].bounding_boxes.append(bbox)
            
    return overlays
