from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, overload, Union, Any, Callable
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
    @overload
    def __init__(self, color: Color) -> None: ...

    def __init__(
        self,
        *args: Union[int, str, Tuple[int, int, int], Color],
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
            elif isinstance(args[0], Color):
                self.r, self.g, self.b = args[0].as_tuple()
            else:
                raise TypeError("Single argument must be tuple[int, int, int] or str or Color.")
        elif len(args) == 3 and all(isinstance(a, int) for a in args):
            self.r, self.g, self.b = args  # type: ignore[assignment]
        else:
            raise TypeError("Invalid arguments for Color constructor.")

    def as_tuple(self) -> Tuple[int, int, int]:
        """Returns the color as a tuple."""
        return (self.r, self.g, self.b)  # type: ignore[return-value]


class OverlayItem(ABC):
    """Abstract base class for overlay items."""
    @abstractmethod
    def __init__(self, color: Color = Color(0, 255, 0), *args, **kwargs):
        self.color = color

    @abstractmethod
    def draw(self, frame: np.ndarray):
        """Draws the overlay item on the given frame."""
        pass

    @abstractmethod
    def warp(self, warp_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """Warps the coordinates of the overlay item using a list of functions."""
        pass


class BoundingBox(OverlayItem):
    """A class to represent a single bounding box."""
    def __init__(self, x: int, y: int, width: int, height: int, label: Optional[str] = None, color: Color = Color(0, 255, 0)):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        self.label = str(label)
        self.color = color.as_tuple() if isinstance(color, Color) else color

    def draw(self, frame: np.ndarray):
        """Draws the bounding box on a frame."""
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, 2)
        if self.label:
            cv2.putText(frame, self.label, (self.x, self.y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, self.color, 1, cv2.LINE_AA)

    def warp(self, warp_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """Warps the bounding box coordinates."""
        # Bounding box is defined by top-left (x, y) and bottom-right (x+w, y+h)
        coords = np.array([[self.x, self.y], [self.x + self.width, self.y + self.height]], dtype=np.float32)
        for func in warp_funcs:
            coords = func(coords)
        
        # After warping, recalculate x, y, width, and height
        self.x, self.y = int(coords[0][0]), int(coords[0][1])
        self.width = int(coords[1][0] - coords[0][0])
        self.height = int(coords[1][1] - coords[0][1])

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height}, label={self.label}, color={self.color})"


class Point(OverlayItem):
    """A class to represent a single point."""
    def __init__(self, x: int, y: int, color: Color = Color(0, 255, 0)):
        self.x = int(x)
        self.y = int(y)
        self.color = color.as_tuple() if isinstance(color, Color) else color

    def draw(self, frame: np.ndarray):
        """Draws the point on a frame."""
        cv2.circle(frame, (self.x, self.y), min(frame.shape[:2]) // 200,  self.color, -1)

    def warp(self, warp_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """Warps the point coordinates."""
        coords = np.array([[self.x, self.y]], dtype=np.float32)
        for func in warp_funcs:
            coords = func(coords)
        self.x, self.y = int(coords[0][0]), int(coords[0][1])

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

    def warp(self, warp_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """Warps the line coordinates."""
        coords = np.array(self.points, dtype=np.float32)
        for func in warp_funcs:
            coords = func(coords)
        self.points = [tuple(int(p) for p in point) for point in coords]

    def __repr__(self) -> str:
        return f"Line(start={self.points[0]}, end={self.points[-1]}, num_points={len(self.points)}, color={self.color})"


class Text(OverlayItem):
    """A class to represent a text overlay."""
    def __init__(self, text: str, position: Tuple[int, int], color: Color = Color(255, 0, 255)):
        self.text = text
        self.position = position
        self.color = color.as_tuple() if isinstance(color, Color) else color

    def draw(self, frame: np.ndarray):
        """Draws the text on a frame."""
        cv2.putText(frame, self.text, self.position, cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2, cv2.LINE_AA)

    def warp(self, warp_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """Warps the text position."""
        coords = np.array([self.position], dtype=np.float32)
        for func in warp_funcs:
            coords = func(coords)
        self.position = (int(coords[0][0]), int(coords[0][1]))

    def __repr__(self) -> str:
        return f"Text(text={self.text}, position={self.position}, color={self.color})"


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
    def try_to_int(value: Any):
        """Attempts to convert a value to an integer, returning it unchanged if it fails."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    return [
        overlay_item(
            *map(try_to_int, row),
            color=color
        ) for row in (np_array.squeeze() if np_array.ndim > 2 else np_array)
    ]
