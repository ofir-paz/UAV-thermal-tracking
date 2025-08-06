import cv2
import numpy as np
from IPython.display import Video as IPythonVideo, display
import os
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, List, Literal
from video_framework.video import Video


@dataclass
class VideoChunk:
    start: int
    end: int
    frames: List[np.ndarray]

    def __iter__(self):
        yield self.start
        yield self.end
        yield self.frames


class Streamer:
    """
    A class to efficiently stream video frames in chunks with optional overlap.
    """
    def __init__(self, video_path: str):
        """
        Initializes the Streamer with video path.

        Args:
            video_path: Path to the video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = os.path.abspath(video_path)
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.fps

    def _convert_to_frames(self, value: float, unit: Literal["frames", "seconds"]) -> int:
        """Converts a value from seconds to frames if unit is 'seconds'."""
        if unit == 'seconds':
            return round(value * self.fps)
        return int(value)

    def stream(
            self, 
            start: Optional[float] = None, 
            end: Optional[float] = None, 
            chunk_size: Optional[float] = None, 
            overlap: float = 0, 
            unit: Literal["frames", "seconds"] = 'frames'
        ) -> Generator[VideoChunk, None, None]:
        """
        Generates video frames in chunks from a specified start to end point.

        Args:
            start: The starting point (frame index or time in seconds). If None, starts from the beginning.
            end: The ending point (frame index or time in seconds). If None, streams to the end.
            chunk_size: The size of each chunk. If None, streams the entire video as one chunk.
            overlap: The overlap between consecutive chunks.
            unit: The unit for chunk_size and overlap ('frames' or 'seconds').

        Yields:
            A tuple (chunk_start_frame, chunk_end_frame, list_of_frames_in_chunk).
        """
        if unit not in ['frames', 'seconds']:
            raise ValueError("Unit must be 'frames' or 'seconds'.")

        _convert = lambda val: self._convert_to_frames(val, unit)

        chunk_size_frames = _convert(chunk_size) if chunk_size is not None else self.frame_count
        overlap_frames = _convert(overlap)

        if overlap_frames >= chunk_size_frames and chunk_size_frames != self.frame_count:
            raise ValueError("Overlap cannot be greater than or equal to chunk_size (unless chunk_size is None).")

        start_frame = _convert(start) if start is not None else 0
        end_frame = min(_convert(end) if end is not None else self.frame_count, self.frame_count)

        if not (0 <= start_frame < self.frame_count):
            raise ValueError(f"Start frame {start_frame} is out of bounds (0-{self.frame_count-1}).")
        if not (0 < end_frame <= self.frame_count):
            raise ValueError(f"End frame {end_frame} is out of bounds (1-{self.frame_count}).")
        if start_frame >= end_frame:
            raise ValueError("Start point must be before end point.")

        current_pos = start_frame
        while True: # Loop indefinitely, break when no more frames or end_frame reached
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            
            chunk_frames = []
            chunk_start_frame = current_pos
            
            # Calculate how many frames to read in this chunk
            # This should not exceed chunk_size_frames, and also not exceed end_frame
            frames_to_read = min(chunk_size_frames, end_frame - current_pos)

            if frames_to_read <= 0: # No more frames to read in this range
                break

            for _ in range(frames_to_read):
                ret, frame = self._cap.read()
                if not ret:
                    # This means we've reached the actual end of the video file
                    # or an error occurred. Break the loop.
                    break 
                chunk_frames.append(frame)
            
            if not chunk_frames: # If no frames were read (e.g., at end of video)
                break

            chunk_end_frame = chunk_start_frame + len(chunk_frames)
            yield VideoChunk(start=chunk_start_frame, end=chunk_end_frame, frames=chunk_frames)
            
            # Determine the start of the next chunk
            if chunk_size_frames == self.frame_count: # If streaming as one chunk, break after first yield
                break
            
            # The next chunk should start after the current chunk, minus the overlap
            next_chunk_start = chunk_start_frame + frames_to_read - overlap_frames
            
            # Ensure next_chunk_start doesn't go before the original start_frame
            if next_chunk_start < start_frame:
                next_chunk_start = start_frame
            
            current_pos = next_chunk_start
            
            # If the last yielded chunk already reached or passed the end_frame, then we are done.
            # This is the crucial part to prevent re-yielding the same last chunk.
            if chunk_end_frame >= end_frame:
                break

    def to_video(self) -> Video:
        """
        Returns a Video object from the video_framework for the entire video.
        """
        return Video(self.video_path)

    def _repr_html_(self):
        """
        Returns the HTML representation for displaying the video in a Jupyter Notebook.
        """
        return IPythonVideo(self.video_path, embed=True)._repr_html_()

    def __repr__(self):
        """
        Displays the original video in a Jupyter Notebook.
        """
        return f"<Streamer video_path='{self.video_path}'>"

    def __del__(self):
        """
        Ensures the video capture object is released when the Streamer object is deleted.
        """
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
