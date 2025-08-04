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

class JupyterPlayer:
    """A class to play a video in a Jupyter Notebook with interactive controls."""

    def __init__(self, video: Video, output_dir: str = "output"):
        self.video = video
        self.output_dir = output_dir
        self.playing = False
        self.current_frame_index = 0
        os.makedirs(self.output_dir, exist_ok=True)

        # Widgets for video display and controls
        self.image_widget = widgets.Image(format='jpeg')
        self.play_button = widgets.Button(description="Play")
        self.pause_button = widgets.Button(description="Pause")
        self.save_button = widgets.Button(description="Save Frame")
        self.progress_slider = widgets.IntSlider(min=0, max=self.video.frame_count - 1, step=1, value=0, description='Frame')
        
        # Widgets for toggling transformations
        self.transform_checkboxes = {}
        for name in self.video.transforms.keys():
            checkbox = widgets.Checkbox(value=True, description=f'Transform: {name}')
            checkbox.observe(self._on_transform_toggle, names='value')
            self.transform_checkboxes[name] = checkbox

        # Widgets for toggling overlays
        self.overlay_checkboxes = {}
        # Collect all unique overlay names from all frames
        all_overlay_names = set()
        for frame_overlays in self.video.overlays.values():
            for name in frame_overlays.keys():
                all_overlay_names.add(name)

        for name in sorted(list(all_overlay_names)):
            checkbox = widgets.Checkbox(value=True, description=f'Overlay: {name}')
            checkbox.observe(self._on_overlay_toggle, names='value')
            self.overlay_checkboxes[name] = checkbox

        # Connect widget events to handlers
        self.play_button.on_click(self._play)
        self.pause_button.on_click(self._pause)
        self.save_button.on_click(self._save_frame)
        self.progress_slider.observe(self._seek, names='value')

        # Layout widgets
        controls = widgets.HBox([self.play_button, self.pause_button, self.save_button])
        transform_toggles = widgets.VBox(list(self.transform_checkboxes.values()))
        overlay_toggles = widgets.VBox(list(self.overlay_checkboxes.values()))
        toggles_box = widgets.HBox([transform_toggles, overlay_toggles])

        self.container = widgets.VBox([self.image_widget, self.progress_slider, controls, toggles_box])

    def _on_transform_toggle(self, change):
        name = change.owner.description.replace('Transform: ', '')
        self.video.set_transform_active(name, change.new)
        self._update_frame()

    def _on_overlay_toggle(self, change):
        name = change.owner.description.replace('Overlay: ', '')
        self.video.set_overlay_active(name, change.new)
        self._update_frame()

    def _play(self, _):
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self._stream_video)
            self.thread.start()

    def _pause(self, _):
        self.playing = False

    def _save_frame(self, _):
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

    def _seek(self, change):
        self.current_frame_index = change.new
        self._update_frame()

    def _update_frame(self):
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        ret, frame = self.video.cap.read()
        if ret:
            processed_frame = frame.copy()
            for transform_name in self.video.active_transforms:
                if transform_name in self.video.transforms:
                    processed_frame = self.video.transforms[transform_name](processed_frame)
            
            if self.current_frame_index in self.video.overlays:
                for overlay_name in self.video.active_overlays:
                    if overlay_name in self.video.overlays[self.current_frame_index]:
                        processed_frame = self.video.overlays[self.current_frame_index][overlay_name].apply(processed_frame)

            # Convert to JPEG for display
            is_success, im_buf_arr = cv2.imencode(".jpg", processed_frame)
            if is_success:
                self.image_widget.value = im_buf_arr.tobytes()

    def _stream_video(self):
        while self.playing and self.current_frame_index < self.video.frame_count - 1:
            self.current_frame_index += 1
            self.progress_slider.value = self.current_frame_index
            # _update_frame is called by the observer of progress_slider.value
            time.sleep(1 / self.video.fps)
        self.playing = False

    def show(self):
        """Displays the player in the notebook."""
        display(self.container)
        self._update_frame()
