from typing import List, Tuple, Optional
import numpy as np
import cv2

class BoundingBox:
    """A class to represent a single bounding box."""
    def __init__(self, x: int, y: int, width: int, height: int, label: Optional[str] = None, color: Tuple[int, int, int] = (0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color

    def draw(self, frame: np.ndarray):
        """Draws the bounding box on a frame."""
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, 2)
        if self.label:
            cv2.putText(frame, self.label, (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)

class Overlay:
    """A class to manage a collection of bounding boxes for a single frame."""
    def __init__(self, bounding_boxes: List[BoundingBox]):
        self.bounding_boxes = bounding_boxes

    def apply(self, frame: np.ndarray):
        """Applies the overlay to a frame."""
        for bbox in self.bounding_boxes:
            bbox.draw(frame)
        return frame