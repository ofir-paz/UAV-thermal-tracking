import cv2
import numpy as np
import os
from typing import Callable, List, Optional, Dict, Tuple
from .overlays import Overlay

class Video:
    """A class to represent a video file, with methods for processing and displaying it."""

    def __init__(self, video_path: str):
        """
        Initializes the Video object.

        Args:
            video_path: The path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self.active_transforms: List[str] = []
        self.overlays: Dict[int, Dict[str, Overlay]] = {}
        self.active_overlays: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Releases the video capture object."""
        if self.cap.isOpened():
            self.cap.release()

    def __iter__(self):
        """Allows iterating over the frames of the video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """Returns the next frame of the video and its corresponding active overlays."""
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        processed_frame = frame.copy()

        for transform_name in self.active_transforms:
            if transform_name in self.transforms:
                processed_frame = self.transforms[transform_name](processed_frame)

        active_frame_overlays = {}
        if frame_number in self.overlays:
            for overlay_name in self.active_overlays:
                if overlay_name in self.overlays[frame_number]:
                    active_frame_overlays[overlay_name] = self.overlays[frame_number][overlay_name]

        return processed_frame, active_frame_overlays

    def add_transform(self, name: str, transform_func: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a transformation function to be applied to each frame.

        Args:
            name: A unique name for the transformation.
            transform_func: A function that takes a frame (NumPy array) and returns a transformed frame.
        """
        self.transforms[name] = transform_func
        self.active_transforms.append(name)

    def set_transform_active(self, name: str, active: bool):
        """
        Sets the active state of a transformation.
        """
        if name in self.transforms:
            if active and name not in self.active_transforms:
                self.active_transforms.append(name)
            elif not active and name in self.active_transforms:
                self.active_transforms.remove(name)

    def add_overlay_to_frame(self, frame_number: int, overlay: Overlay):
        """
        Adds an overlay to be applied to a specific frame.

        Args:
            frame_number: The frame number to apply the overlay to.
            overlay: An Overlay object.
        """
        if frame_number not in self.overlays:
            self.overlays[frame_number] = {}
        self.overlays[frame_number][overlay.name] = overlay
        if overlay.name not in self.active_overlays:
            self.active_overlays.append(overlay.name)

    def add_overlays(self, overlays_by_frame: Dict[int, Dict[str, Overlay]]):
        """
        Adds a dictionary of overlays, mapping frame numbers to dictionaries of Overlay objects.
        """
        for frame_number, frame_overlays in overlays_by_frame.items():
            if frame_number not in self.overlays:
                self.overlays[frame_number] = {}
            self.overlays[frame_number].update(frame_overlays)
            for overlay_name in frame_overlays.keys():
                if overlay_name not in self.active_overlays:
                    self.active_overlays.append(overlay_name)

    def set_overlay_active(self, name: str, active: bool):
        """
        Sets the active state of an overlay.
        """
        if active and name not in self.active_overlays:
            self.active_overlays.append(name)
        elif not active and name in self.active_overlays:
            self.active_overlays.remove(name)

    def save_frames_where(self, predicate: Callable[[np.ndarray], bool], output_dir: str = "output"):
        """
        Saves frames from the video that satisfy a given predicate.
        The predicate is applied on the transformed frame. Active overlays are applied to the saved image.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = frame.copy()
            for transform_name in self.active_transforms:
                if transform_name in self.transforms:
                    processed_frame = self.transforms[transform_name](processed_frame)

            if predicate(processed_frame):
                output_frame = processed_frame.copy()
                if frame_idx in self.overlays:
                    for overlay_name in self.active_overlays:
                        if overlay_name in self.overlays[frame_idx]:
                            output_frame = self.overlays[frame_idx][overlay_name].apply(output_frame)

                filepath = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(filepath, output_frame)
                print(f"Saved frame {frame_idx} to {filepath}")
            frame_idx += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
