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
        self.lock = threading.Lock() # To protect shared resources

    def _update_frame(self):
        with self.lock:
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
                self.current_frame = processed_frame
                cv2.imshow(self.window_name, self.current_frame)
                cv2.setTrackbarPos(self.trackbar_name, self.window_name, self.current_frame_index)

    def _stream_video(self):
        while self.playing and self.current_frame_index < self.video.frame_count - 1:
            self.current_frame_index += 1
            # _update_frame is now called by the main loop
            time.sleep(1 / self.video.fps)
        self.playing = False

    def _on_trackbar_change(self, frame_pos):
        if abs(self.current_frame_index - frame_pos) > 1 or not self.playing: # Only seek if significant change or not playing
            self._seek(frame_pos)

    def show(self):
        """Displays the video in a desktop window with controls."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.trackbar_name, self.window_name, 0, self.video.frame_count - 1, self._on_trackbar_change)

        self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        self._update_frame()

        print(f"\nDesktop Player Controls:\n")
        print(f"  - Spacebar: Play/Pause\n")
        print(f"  - S: Save current frame\n")
        print(f"  - Q: Quit\n")
        print(f"  - Use the trackbar to seek frames\n")

        while True:
            if self.playing:
                self.current_frame_index += 1
                if self.current_frame_index >= self.video.frame_count:
                    self.current_frame_index = self.video.frame_count - 1
                    self.playing = False
                self._update_frame()

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

        cv2.destroyAllWindows()
        self.video.release()
