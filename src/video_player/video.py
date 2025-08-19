import cv2
import numpy as np
import os
from typing import Callable, List, Optional, Dict, Tuple, Any, Union
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
        self.state: Dict[Any, Any] = {}
        self.play_mode = 'processed'  # 'processed' or 'original_with_remapped'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Releases the video capture object."""
        if self.cap.isOpened():
            self.cap.release()

    def set_play_mode(self, mode: str):
        """Sets the play mode for the video."""
        if mode not in ['processed', 'original_with_remapped']:
            raise ValueError("Invalid play mode specified.")
        self.play_mode = mode

    def __iter__(self):
        """Allows iterating over the frames of the video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """Returns the next frame of the video and its corresponding active overlays."""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        processed_frame, overlays = self.get_frame(frame_number, frame)
        return processed_frame, overlays

    def add_transform(self, name: str, transform_func: Union[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, Dict[Any, Any]], np.ndarray]]):
        """
        Adds a transformation function to be applied to each frame.
        The function can optionally return a warp function for coordinate remapping.
        """
        self.operations.append((name, 'transform', transform_func))
        self.active_operations.append(name)

    def add_online_overlay(self, name: str, overlay_func: Union[Callable[[np.ndarray], List[OverlayItem]], Callable[[np.ndarray, Dict[Any, Any]], List[OverlayItem]]]):
        """
        Adds an online overlay to be generated on the fly.
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

    def _process_frame_processed_mode(self, frame: np.ndarray, frame_index: int) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """
        Applies transformations and overlays to a given frame for 'processed' mode.
        """
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        processed_frame = frame
        online_overlay_items = []

        for name, op_type, func in self.operations:
            if name in self.active_operations:
                if op_type == 'transform':
                    try:
                        result = func(processed_frame, self.state)
                    except TypeError:
                        result = func(processed_frame)
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        processed_frame, _ = result
                    else:
                        processed_frame = result

                elif op_type == 'online_overlay':
                    try:
                        new_overlay_items = func(processed_frame, self.state)
                    except TypeError:
                        new_overlay_items = func(processed_frame)
                    online_overlay_items.extend(new_overlay_items)

        active_frame_overlays = {}
        if frame_index in self.overlays:
            for overlay_name in self.active_overlays:
                if overlay_name in self.overlays[frame_index]:
                    active_frame_overlays[overlay_name] = self.overlays[frame_index][overlay_name]
        
        if online_overlay_items:
            active_frame_overlays['online_overlays'] = Overlay(name='online_overlays', overlay_items=online_overlay_items)

        if len(processed_frame.shape) == 2 or processed_frame.shape[2] == 1:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        for overlay in active_frame_overlays.values():
            processed_frame = overlay.apply(processed_frame)

        return processed_frame, active_frame_overlays

    def _get_original_with_remapped_overlays(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Gets the original frame and applies overlays remapped from the transformed space.
        """
        original_frame = frame.copy()
        processed_frame = frame
        if self.grayscale and len(processed_frame.shape) == 3:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        warp_funcs = []
        all_overlay_items = []

        # Handle online overlays
        for name, op_type, func in self.operations:
            if name in self.active_operations:
                if op_type == 'transform':
                    try:
                        result = func(processed_frame, self.state)
                    except TypeError:
                        result = func(processed_frame)
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        processed_frame, warp_func = result
                        if warp_func:
                            warp_funcs.append(warp_func)
                    else:
                        processed_frame = result
                
                elif op_type == 'online_overlay':
                    try:
                        new_overlay_items = func(processed_frame, self.state)
                    except TypeError:
                        new_overlay_items = func(processed_frame)
                    
                    if warp_funcs:
                        for item in new_overlay_items:
                            item.warp(list(reversed(warp_funcs)))
                    
                    all_overlay_items.extend(new_overlay_items)

        # Handle offline overlays
        if frame_index in self.overlays:
            for overlay_name in self.active_overlays:
                if overlay_name in self.overlays[frame_index]:
                    offline_overlay = self.overlays[frame_index][overlay_name]
                    all_overlay_items.extend(offline_overlay.overlay_items)

        # Apply remapped overlays to the original frame
        final_frame = original_frame
        if len(final_frame.shape) == 2:
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_GRAY2BGR)
            
        remapped_overlay = Overlay(name="remapped", overlay_items=all_overlay_items)
        final_frame = remapped_overlay.apply(final_frame)

        return final_frame

    def get_frame(self, frame_index: int, frame: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Overlay]]:
        """
        Gets a specific frame by seeking, applies transformations and overlays.
        """
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                raise IndexError("Frame index out of range")

        if self.play_mode == 'original_with_remapped':
            processed_frame = self._get_original_with_remapped_overlays(frame, frame_index)
            return processed_frame, {}
        else:
            processed_frame, overlays = self._process_frame_processed_mode(frame, frame_index)
            return processed_frame, overlays

    def save_frames_where(self, predicate: Optional[Callable[[np.ndarray], bool]] = None, output_dir: str = "output"):
        """
        Saves frames from the video that satisfy a given predicate.
        """
        predicate = predicate or (lambda x: True)
        os.makedirs(output_dir, exist_ok=True)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in tqdm(range(self.frame_count), desc="Processing and saving frames"):
            processed_frame, _ = self.get_frame(frame_idx)
            if predicate(processed_frame):
                filepath = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(filepath, processed_frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save_video(self, output_path: str, codec: str = 'mp4v', fps: Optional[float] = None):
        """
        Saves the video with applied transformations and overlays to a file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if fps is None:
            fps = self.fps

        fourcc = cv2.VideoWriter.fourcc(*codec)
        
        try:
            processed_frame, _ = self.get_frame(0)
        except IndexError:
            print("Video has no frames.")
            return

        height, width, _ = processed_frame.shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in tqdm(range(self.frame_count), desc="Saving video"):
            processed_frame, _ = self.get_frame(frame_idx)
            out.write(processed_frame)

        out.release()
        print(f"Video saved to {output_path}")
