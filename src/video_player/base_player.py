import threading
import time
import cv2
import os
import numpy as np
from abc import ABC, abstractmethod

from .video import Video
from .overlays import Overlay

class BasePlayer(ABC):
    """Abstract base class for video players."""

    def __init__(self, video: Video, output_dir: str = "output"):
        self.video = video
        self.output_dir = output_dir
        self.playing = False
        self.current_frame_index = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def _play(self):
        self.playing = True

    def _pause(self):
        self.playing = False

    def _save_frame(self):
        # Ensure the video capture is at the correct frame before saving
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.video.cap.read()
        if ret:
            # Apply transformations and overlays before saving
            processed_frame = frame.copy()
            for transform_name in self.video.active_transforms:
                if transform_name in self.video.transforms:
                    processed_frame = self.video.transforms[transform_name](processed_frame)
            
            if self.current_frame_index in self.video.overlays:
                for overlay_name in self.video.active_overlays:
                    if overlay_name in self.video.overlays[self.current_frame_index]:
                        processed_frame = self.video.overlays[self.current_frame_index][overlay_name].apply(processed_frame)

            filepath = os.path.join(self.output_dir, f"frame_{self.current_frame_index}.jpg")
            cv2.imwrite(filepath, processed_frame)
            print(f"Frame {self.current_frame_index} saved to {filepath}")

    def _seek(self, frame_index: int):
        self.current_frame_index = frame_index
        self._update_frame()

    def _on_transform_toggle(self, name: str, active: bool):
        self.video.set_transform_active(name, active)
        self._update_frame()

    def _on_overlay_toggle(self, name: str, active: bool):
        self.video.set_overlay_active(name, active)
        self._update_frame()

    @abstractmethod
    def _update_frame(self):
        """Abstract method to update the displayed frame. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _stream_video(self):
        """Abstract method for video streaming logic. Implemented by subclasses."""
        pass

    @abstractmethod
    def show(self):
        """Abstract method to display the player. Must be implemented by subclasses."""
        pass
