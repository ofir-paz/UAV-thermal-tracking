from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, overload, Union
import numpy as np
import cv2


class Color:
    """A simple class to hold RGB color values."""

    @overload
    def __init__(self, color: Tuple[int, int, int]) -> None: ...
    @overload
    def __init__(self, r: int, g: int, b: int) -> None: ...
    @overload
    def __init__(self, color: str) -> None: ...

    def __init__(
        self,
        *args: Union[int, str, Tuple[int, int, int]],
    ) -> None:
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, tuple):
                self.r, self.g, self.b = arg
            elif isinstance(arg, str):
                if arg.startswith('#'):
                    hex_str = arg.lstrip('#')
                    self.r = int(hex_str[0:2], 16)
                    self.g = int(hex_str[2:4], 16)
                    self.b = int(hex_str[4:6], 16)
                else:
                    raise ValueError(
                        "Color string must start with '#' or provide an RGB tuple."
                    )
            else:
                raise TypeError("Single argument must be tuple[int, int, int] or str.")
        elif len(args) == 3 and all(isinstance(a, int) for a in args):
            self.r, self.g, self.b = args  # type: ignore[assignment]
        else:
            raise TypeError("Invalid arguments for Color constructor.")

    def as_tuple(self) -> Tuple[int, int, int]:
        """Returns the color as a tuple."""
        return (self.r, self.g, self.b)


class OverlayItem(ABC):
    """Abstract base class for overlay items."""
    @abstractmethod
    def __init__(self, color: Color = Color(0, 255, 0), *args, **kwargs):
        self.color = color

    @abstractmethod
    def draw(self, frame: np.ndarray):
        """Draws the overlay item on the given frame."""
        pass


class BoundingBox(OverlayItem):
    """A class to represent a single bounding box."""
    def __init__(self, x: int, y: int, width: int, height: int, label: Optional[str] = None, color: Color = Color(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color.as_tuple() if isinstance(color, Color) else color

    def draw(self, frame: np.ndarray):
        """Draws the bounding box on a frame."""
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, 2)
        if self.label:
            cv2.putText(frame, self.label, (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height}, label={self.label}, color={self.color})"


class Point(OverlayItem):
    """A class to represent a single point."""
    def __init__(self, x: int, y: int, color: Color = Color(0, 255, 0)):
        self.x = x
        self.y = y
        self.color = color.as_tuple() if isinstance(color, Color) else color

    def draw(self, frame: np.ndarray):
        """Draws the point on a frame."""
        cv2.circle(frame, (self.x, self.y), min(frame.shape[:2]) // 200,  self.color, -1)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, color={self.color})"


class Line(OverlayItem):
    """
    A polyline defined by an arbitrary-length list of (x, y) points.
    Drawn open (not closed).
    """
    def __init__(
        self,
        points: List[Tuple[int, int]],
        color: Color = Color(0, 255, 0),
        thickness: int = 2,
    ):
        assert len(points) >= 2, "Line needs at least two points"
        self.points = points
        self.color = color.as_tuple() if isinstance(color, Color) else color
        self.thickness = thickness
    
    def extend(self, point: Tuple[int, int]):
        """Adds a point to the line."""
        self.points.append(point)

    def draw(self, frame: np.ndarray):
        pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=self.color, thickness=self.thickness)
        cv2.circle(frame, self.points[-1], min(frame.shape[:2]) // 200, self.color, -1)

    def __repr__(self) -> str:
        return f"Line(start={self.points[0]}, end={self.points[-1]}, num_points={len(self.points)}, color={self.color})"


class Overlay:
    """A class to manage a collection of bounding boxes for a single frame."""
    def __init__(self, name: str, overlay_items: List[OverlayItem]):
        self.name = name
        self.overlay_items = overlay_items

    def apply(self, frame: np.ndarray):
        """Applies the overlay to a frame."""
        for item in self.overlay_items:
            item.draw(frame)
        return frame
    

def np_to_overlay_items(
    np_array: np.ndarray, 
    overlay_item: Type[OverlayItem], 
    color: Color = Color(0, 255, 0)
) -> List[OverlayItem]:
    """
    Converts a NumPy array of overlay item parameters to a list of OverlayItem objects.
    
    Args:
        np_array: A NumPy array where each row represents an overlay item in the format [x, y, width, height].
        color: The color to use for the overlay items.

    Returns:
        A list of OverlayItem objects representing the overlay items.
    """
    return [
        overlay_item(
            *map(int, row),
            color=color
        ) for row in np_array.squeeze()
    ]
