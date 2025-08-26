import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import io
import threading
import time
import cv2
import os
import numpy as np

from .video import Video
from .overlays import Overlay
from .base_player import BasePlayer

class JupyterPlayer(BasePlayer):
    """A class to play a video in a Jupyter Notebook with interactive controls."""

    def __init__(self, video: Video, output_dir: str = "output"):
        super().__init__(video, output_dir)
        
        # Widgets for video display and controls
        self.image_widget = widgets.Image(format='jpeg')
        self.play_button = widgets.Button(description="Play")
        self.pause_button = widgets.Button(description="Pause")
        self.save_button = widgets.Button(description="Save Frame")
        self.progress_slider = widgets.IntSlider(min=0, max=self.video.frame_count - 1, step=1, value=0, description='Frame')
        
        # Widgets for toggling operations
        self.operation_checkboxes = {}
        for name, op_type, _ in self.video.operations:
            checkbox = widgets.Checkbox(value=True, description=f'{op_type.replace("_", " ").title()}: {name}')
            checkbox.observe(self._on_operation_toggle_wrapper, names='value')
            self.operation_checkboxes[name] = checkbox

        # Widgets for toggling overlays
        self.overlay_checkboxes = {}
        # Collect all unique overlay names from all frames
        all_overlay_names = set()
        for frame_overlays in self.video.overlays.values():
            for name in frame_overlays.keys():
                all_overlay_names.add(name)

        for name in sorted(list(all_overlay_names)):
            checkbox = widgets.Checkbox(value=True, description=f'Overlay: {name}')
            checkbox.observe(self._on_overlay_toggle_wrapper, names='value')
            self.overlay_checkboxes[name] = checkbox

        # Connect widget events to handlers
        self.play_button.on_click(lambda x: self._play())
        self.pause_button.on_click(lambda x: self._pause())
        self.save_button.on_click(lambda x: self._save_frame())
        self.progress_slider.observe(self._seek_wrapper, names='value')

        # Layout widgets
        controls = widgets.HBox([self.play_button, self.pause_button, self.save_button])
        operation_toggles = widgets.VBox(list(self.operation_checkboxes.values()))
        overlay_toggles = widgets.VBox(list(self.overlay_checkboxes.values()))
        toggles_box = widgets.HBox([operation_toggles, overlay_toggles])

        self.container = widgets.VBox([self.image_widget, self.progress_slider, controls, toggles_box])

    def _play(self):
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self._stream_video)
            self.thread.start()

    def _pause(self):
        self.playing = False

    def _on_operation_toggle_wrapper(self, change):
        description = change.owner.description
        name = description.split(": ", 1)[1]
        self._on_operation_toggle(name, change.new)

    def _on_overlay_toggle_wrapper(self, change):
        name = change.owner.description.replace('Overlay: ', '')
        self._on_overlay_toggle(name, change.new)

    def _seek_wrapper(self, change):
        self._seek(change.new)

    def _update_frame(self, processed_frame: np.ndarray):
        # Display real-time FPS
        cv2.putText(processed_frame, f"FPS: {self.real_time_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert to JPEG for display
        is_success, im_buf_arr = cv2.imencode(".jpg", processed_frame)
        if is_success:
            self.image_widget.value = im_buf_arr.tobytes()

    def _stream_video(self):
        # Reset video to the beginning for sequential reading if not already playing from current position
        if self.current_frame_index != int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1:
            self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

        while self.playing:
            start_time = time.time()
            try:
                processed_frame, _ = next(self.video)
                self.current_frame_index = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.progress_slider.value = self.current_frame_index
                self._update_frame(processed_frame)
            except StopIteration:
                self.playing = False
                self.current_frame_index = self.video.frame_count - 1
                # Get the last frame to display it
                processed_frame, _ = self.video.get_frame(self.current_frame_index)
                self._update_frame(processed_frame)
                break

            end_time = time.time()
            self.frame_times.append(end_time - start_time)
            if len(self.frame_times) > 1:
                self.real_time_fps = len(self.frame_times) / sum(self.frame_times)

            # Adjust sleep time to maintain desired FPS
            elapsed_time = time.time() - start_time
            sleep_time = (1 / self.video.fps) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If processing takes longer than 1/fps, don't sleep, but ensure thread yields
                time.sleep(0.001) # Small sleep to yield control

    def show(self):
        """Displays the player in the notebook."""
        display(self.container)
        processed_frame, _ = self.video.get_frame(self.current_frame_index)
        self._update_frame(processed_frame)
