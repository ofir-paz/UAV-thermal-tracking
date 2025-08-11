import threading
import time
import cv2
import os
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

from .video import Video
from .overlays import Overlay

class BasePlayer(ABC):
    """Abstract base class for video players."""

    def __init__(self, video: Video, output_dir: str = "output"):
        self.video = video
        self.output_dir = output_dir
        self.playing = False
        self.current_frame_index = 0
        self.real_time_fps = 0
        self.frame_times = deque(maxlen=60)

    def _play(self):
        self.playing = True

    def _pause(self):
        self.playing = False
        self.frame_times.clear()

    def _save_frame(self):
        os.makedirs(self.output_dir, exist_ok=True)
        processed_frame = self.video.get_frame(self.current_frame_index)
        filepath = os.path.join(self.output_dir, f"frame_{self.current_frame_index}.jpg")
        cv2.imwrite(filepath, processed_frame)
        print(f"Frame {self.current_frame_index} saved to {filepath}")

    def _seek(self, frame_index: int):
        self.current_frame_index = frame_index
        processed_frame, _ = self.video.get_frame(self.current_frame_index)
        self._update_frame(processed_frame)

    def _on_operation_toggle(self, name: str, active: bool):
        self.video.set_operation_active(name, active)
        processed_frame, _ = self.video.get_frame(self.current_frame_index)
        self._update_frame(processed_frame)

    def _on_overlay_toggle(self, name: str, active: bool):
        self.video.set_overlay_active(name, active)
        processed_frame, _ = self.video.get_frame(self.current_frame_index)
        self._update_frame(processed_frame)

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
