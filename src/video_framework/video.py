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

        self.transforms: List[Callable[[np.ndarray], np.ndarray]] = []
        self.overlays: Dict[int, Overlay] = {}

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

    def __next__(self) -> Tuple[np.ndarray, Optional[Overlay]]:
        """Returns the next frame of the video and its corresponding overlay."""
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        processed_frame = frame.copy()

        for transform in self.transforms:
            processed_frame = transform(processed_frame)

        overlay = self.overlays.get(frame_number)

        return processed_frame, overlay

    def add_transform(self, transform: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a transformation to be applied to each frame.

        Args:
            transform: A function that takes a frame (NumPy array) and returns a transformed frame.
        """
        self.transforms.append(transform)

    def add_overlay_to_frame(self, frame_number: int, overlay: Overlay):
        """
        Adds an overlay to be applied to a specific frame.

        Args:
            frame_number: The frame number to apply the overlay to.
            overlay: An Overlay object.
        """
        self.overlays[frame_number] = overlay

    def add_overlays(self, overlays: Dict[int, Overlay]):
        """
        Adds a dictionary of overlays, mapping frame numbers to Overlay objects.

        Args:
            overlays: A dictionary of overlays.
        """
        self.overlays.update(overlays)

    def save_frames_where(self, predicate: Callable[[np.ndarray], bool], output_dir: str = "output"):
        """
        Saves frames from the video that satisfy a given predicate.
        The predicate is applied on the transformed frame. Overlays are applied to the saved image.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = frame.copy()
            for transform in self.transforms:
                processed_frame = transform(processed_frame)

            if predicate(processed_frame):
                output_frame = processed_frame.copy()
                overlay = self.overlays.get(frame_idx)
                if overlay:
                    output_frame = overlay.apply(output_frame)

                filepath = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(filepath, output_frame)
                print(f"Saved frame {frame_idx} to {filepath}")
            frame_idx += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)