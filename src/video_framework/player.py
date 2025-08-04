
import cv2
import os
from .video import Video

class Player:
    """A class to play a video with interactive controls."""

    def __init__(self, video: Video, output_dir: str = "output"):
        """
        Initializes the Player object.

        Args:
            video: The Video object to play.
            output_dir: The directory to save output frames.
        """
        self.video = video
        self.window_name = f"Video Player: {self.video.video_path}"
        self.paused = False
        self.show_overlays = True
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def play(self):
        """Plays the video in a new window."""
        cv2.namedWindow(self.window_name)

        frame_iterator = iter(self.video)
        current_frame, overlay = next(frame_iterator, (None, None))

        while current_frame is not None:
            display_frame = current_frame.copy()
            if self.show_overlays and overlay:
                display_frame = overlay.apply(display_frame)

            cv2.imshow(self.window_name, display_frame)

            delay = 0 if self.paused else int(1000 / self.video.fps)
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar to pause/play
                self.paused = not self.paused
            elif key == ord('s'):  # 's' to save frame
                self.save_frame(current_frame)
            elif key == ord('o'):  # 'o' to toggle overlays
                self.show_overlays = not self.show_overlays

            if not self.paused:
                current_frame, overlay = next(frame_iterator, (None, None))

        cv2.destroyAllWindows()

    def save_frame(self, frame):
        """Saves the current frame to a file."""
        frame_number = int(self.video.cap.get(cv2.CAP_PROP_POS_FRAMES))
        filename = f"frame_{frame_number}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Frame {frame_number} saved to {filepath}")
