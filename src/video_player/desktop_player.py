import cv2
import threading
import time
import numpy as np

from .video import Video
from .base_player import BasePlayer

class DesktopPlayer(BasePlayer):
    """A class to play a video in a desktop window."""

    def __init__(self, video: Video, window_name: str = "Video Player", output_dir: str = "output"):
        super().__init__(video, output_dir)
        self.window_name = window_name
        self.current_frame = None
        self.trackbar_name = "Frame"
        self.key_map = {}
        self.next_key_code = ord('1') # Start assigning keys from '1'

    def _get_next_key(self):
        key = chr(self.next_key_code)
        self.next_key_code += 1
        return key

    def _play(self):
        self.playing = True

    def _pause(self):
        self.playing = False

    def _update_frame(self, processed_frame: np.ndarray):
        self.current_frame = processed_frame
            
        # Display real-time FPS
        cv2.putText(self.current_frame, f"FPS: {self.real_time_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.window_name, self.current_frame)
        cv2.setTrackbarPos(self.trackbar_name, self.window_name, self.current_frame_index)

    def _stream_video(self):
        # DesktopPlayer handles streaming in its main loop, no separate thread needed
        pass

    def _on_trackbar_change(self, frame_pos):
        if abs(self.current_frame_index - frame_pos) > 1 or not self.playing: # Only seek if significant change or not playing
            self._seek(frame_pos)

    def show(self):
        """Displays the video in a desktop window with controls."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.trackbar_name, self.window_name, 0, self.video.frame_count - 1, self._on_trackbar_change)

        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        processed_frame, _ = self.video.get_frame(self.current_frame_index)
        self._update_frame(processed_frame)

        print(f"\nDesktop Player Controls:\n")
        print(f"  - Spacebar: Play/Pause\n")
        print(f"  - S: Save current frame\n")
        print(f"  - Q: Quit\n")
        print(f"  - Use the trackbar to seek frames\n")

        print(f"\nOperation Toggles:\n")
        for name, op_type, _ in self.video.operations:
            key = self._get_next_key()
            self.key_map[ord(key)] = ("operation", name)
            print(f"  - {key}: Toggle {op_type.replace('_', ' ').title()} '{name}'\n")

        print(f"\nOverlay Toggles:\n")
        all_overlay_names = set()
        for frame_overlays in self.video.overlays.values():
            for name in frame_overlays.keys():
                all_overlay_names.add(name)
        for name in sorted(list(all_overlay_names)):
            key = self._get_next_key()
            self.key_map[ord(key)] = ("overlay", name)
            print(f"  - {key}: Toggle Overlay '{name}'\n")

        # Reset video to the beginning for sequential reading
        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            if self.playing:
                start_time = time.time()
                try:
                    processed_frame, _ = next(self.video)
                    self.current_frame_index = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self._update_frame(processed_frame)
                except StopIteration:
                    self.playing = False
                    self.current_frame_index = self.video.frame_count - 1
                    processed_frame, _ = self.video.get_frame(self.current_frame_index) # Get last frame
                    self._update_frame(processed_frame)

                end_time = time.time()
                self.frame_times.append(end_time - start_time)
                if len(self.frame_times) > 1:
                    self.real_time_fps = len(self.frame_times) / sum(self.frame_times)

            key = cv2.waitKey(int(1000 / self.video.fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if self.playing:
                    self._pause()
                else:
                    self._play()
            elif key == ord('s'):
                self._save_frame()
            elif key in self.key_map:
                item_type, name = self.key_map[key]
                if item_type == "operation":
                    current_state = name in self.video.active_operations
                    self._on_operation_toggle(name, not current_state)
                elif item_type == "overlay":
                    current_state = name in self.video.active_overlays
                    self._on_overlay_toggle(name, not current_state)

        cv2.destroyAllWindows()
        self.video.release()
