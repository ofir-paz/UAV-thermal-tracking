import cv2
import numpy as np
import os
from typing import Callable, List, Optional, Dict, Tuple
from .overlays import Overlay, OverlayItem
from tqdm import tqdm

class Video:
    """A class to represent a video file, with methods for processing and displaying it."""

    def __init__(self, video_path: str, grayscale: bool = False):
        """
        Initializes the Video object.

        Args:
            video_path: The path to the video file.
            grayscale: Whether to load the video in grayscale.
        """
        self.video_path = video_path
        self.grayscale = grayscale
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.operations: List[Tuple[str, str, Callable]] = []
        self.active_operations: List[str] = []

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
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 # get(CAP_PROP_POS_FRAMES) returns the index of the next frame to be decoded.
        processed_frame, active_frame_overlays = self._process_frame(frame, frame_number)
        
        return processed_frame, active_frame_overlays

    def add_transform(self, name: str, transform_func: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a transformation function to be applied to each frame.

        Args:
            name: A unique name for the transformation.
            transform_func: A function that takes a frame (NumPy array) and returns a transformed frame.
        """
        self.operations.append((name, 'transform', transform_func))
        self.active_operations.append(name)

    def add_online_overlay(self, name: str, overlay_func: Callable[[np.ndarray], List[OverlayItem]]):
        """
        Adds an online overlay to be generated on the fly.

        Args:
            name: A unique name for the online overlay.
            overlay_func: A function that takes a frame and returns a list of OverlayItem objects.
        """
        self.operations.append((name, 'online_overlay', overlay_func))
        self.active_operations.append(name)

    def set_operation_active(self, name: str, active: bool):
        """
        Sets the active state of an operation.
        """
        if any(op[0] for op in self.operations if op[0] == name):
            if active and name not in self.active_operations:
                self.active_operations.append(name)
            elif not active and name in self.active_operations:
                self.active_operations.remove(name)

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

    def _process_frame(self, frame: np.ndarray, frame_index: int) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """
        Applies transformations and overlays to a given frame.
        """
        if self.grayscale and len(frame.shape) == 3: # Only convert if it's a color frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        processed_frame = frame
        online_overlay_items = []

        for name, op_type, func in self.operations:
            if name in self.active_operations:
                if op_type == 'transform':
                    processed_frame = func(processed_frame)
                elif op_type == 'online_overlay':
                    online_overlay_items.extend(func(processed_frame))

        # Ensure the frame is in color before applying overlays
        if len(processed_frame.shape) == 2 or processed_frame.shape[2] == 1:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        active_frame_overlays = {}
        if frame_index in self.overlays:
            for overlay_name in self.active_overlays:
                if overlay_name in self.overlays[frame_index]:
                    active_frame_overlays[overlay_name] = self.overlays[frame_index][overlay_name]
        
        if online_overlay_items:
            active_frame_overlays['online_overlays'] = Overlay(name='online_overlays', overlay_items=online_overlay_items)

        for overlay in active_frame_overlays.values():
            processed_frame = overlay.apply(processed_frame)

        return processed_frame, active_frame_overlays

    def get_frame(self, frame_index: int) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """
        Gets a specific frame by seeking, applies transformations and overlays.
        This method is for random access and should be used sparingly for performance.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Frame index out of range")
        return self._process_frame(frame, frame_index)

    def save_frames_where(self, predicate: Optional[Callable[[np.ndarray], bool]] = None, output_dir: str = "output"):
        """
        Saves frames from the video that satisfy a given predicate.
        The predicate is applied on the transformed frame. Active overlays are applied to the saved image.
        """
        predicate = predicate or (lambda x: True)  # Default to always true if no predicate is provided
        os.makedirs(output_dir, exist_ok=True)

        # Reset video to the beginning for sequential reading
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx, (processed_frame, _) in enumerate(tqdm(self, desc="Processing and saving frames", total=self.frame_count)):
            if predicate(processed_frame):
                filepath = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(filepath, processed_frame)

        # Reset video to the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save_video(self, output_path: str, codec: str = 'mp4v', fps: Optional[float] = None):
        """
        Saves the video with applied transformations and overlays to a file.
        
        Args:
            output_path: The path to save the output video.
            codec: The codec to use for saving the video.
            fps: Frames per second for the output video. Defaults to the original video's FPS.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if fps is None:
            fps = self.fps

        fourcc = cv2.VideoWriter.fourcc(*codec)

        # Reset video to the beginning for sequential reading
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get the first frame to determine the size
        try:
            processed_frame, _ = next(self)
        except StopIteration:
            print("Video has no frames.")
            return

        width, height = processed_frame.shape[1], processed_frame.shape[0]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        out.write(processed_frame)  # Write the first frame
        
        for processed_frame, _ in tqdm(self, desc="Saving video", total=self.frame_count - 1):
            out.write(processed_frame)

        out.release()
        print(f"Video saved to {output_path}")
