import cv2
import numpy as np
from IPython.display import Video as IPythonVideo, display
import os
from dataclasses import dataclass
from typing import Optional, Generator, List, Literal
from collections import deque
from video_player.video import Video


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
    def __init__(self, video_path: str, grayscale: bool = False):
        """
        Initializes the Streamer with video path.

        Args:
            video_path: Path to the video file.
            grayscale: Whether to load the video in grayscale.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = os.path.abspath(video_path)
        self.grayscale = grayscale
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.fps

    def _read_frames_batch(self, num_frames: int) -> List[np.ndarray]:
        """Reads a batch of frames from the video capture object."""
        frames = []
        for _ in range(num_frames):
            ret, frame = self._cap.read()
            if not ret:
                break
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        return frames

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
            A VideoChunk object containing start, end, and frames.
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

        # Initialize cache and prefetch buffer
        self._frame_cache = deque() # Stores frames from the previous chunk that overlap with the current one
        self._prefetch_buffer = deque() # Stores frames read ahead of time
        self._current_prefetch_pos = start_frame # Tracks the position up to which frames have been prefetched

        def _fill_prefetch_buffer(target_buffer_size: int):
            """Fills the prefetch buffer up to target_buffer_size frames."""
            frames_to_read_into_buffer = target_buffer_size - len(self._prefetch_buffer)
            if frames_to_read_into_buffer > 0:
                # Move the capture position to where we left off prefetching
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_prefetch_pos)
                new_frames = self._read_frames_batch(frames_to_read_into_buffer)
                self._prefetch_buffer.extend(new_frames)
                self._current_prefetch_pos += len(new_frames)

        # Initial prefetch: fill buffer with at least prefetch_factor chunks
        _fill_prefetch_buffer(chunk_size_frames * 2) # Default prefetch_factor is 2

        current_pos = start_frame
        while True: # Loop indefinitely, break when no more frames or end_frame reached
            chunk_frames = []
            chunk_start_frame = current_pos
            
            # 1. Get frames from cache (overlap from previous chunk)
            frames_from_cache = []
            if overlap_frames > 0 and len(self._frame_cache) >= overlap_frames:
                frames_from_cache = [self._frame_cache.popleft() for _ in range(overlap_frames)]
            chunk_frames.extend(frames_from_cache)

            # 2. Determine how many new frames are needed for this chunk
            new_frames_needed = chunk_size_frames - len(chunk_frames)
            
            # Ensure we don't read beyond end_frame
            new_frames_needed = min(new_frames_needed, end_frame - current_pos - len(frames_from_cache))

            if new_frames_needed <= 0 and not chunk_frames: # No more frames to read or yield
                break

            # 3. Get new frames from prefetch buffer
            new_frames = []
            if new_frames_needed > 0:
                # Ensure prefetch buffer has enough frames
                if len(self._prefetch_buffer) < new_frames_needed:
                    _fill_prefetch_buffer(len(self._prefetch_buffer) + new_frames_needed) # Fill up to what's needed

                # Take frames from prefetch buffer
                for _ in range(new_frames_needed):
                    if self._prefetch_buffer:
                        new_frames.append(self._prefetch_buffer.popleft())
                    else:
                        break # Ran out of prefetched frames
            chunk_frames.extend(new_frames)
            
            if not chunk_frames: # If no frames were read (e.g., at end of video)
                break

            chunk_end_frame = chunk_start_frame + len(chunk_frames)
            
            yield VideoChunk(start=chunk_start_frame, end=chunk_end_frame, frames=chunk_frames)

            # 4. Update cache for the next iteration (store overlapping frames from current chunk)
            self._frame_cache.clear() # Clear previous cache
            if overlap_frames > 0 and len(chunk_frames) >= overlap_frames:
                self._frame_cache.extend(chunk_frames[len(chunk_frames) - overlap_frames:])

            # 5. Advance current_pos for the next chunk
            current_pos = chunk_start_frame + len(chunk_frames) - overlap_frames
            if current_pos < start_frame:
                current_pos = start_frame

            # 6. Refill prefetch buffer if it's getting low (maintain prefetch_factor)
            _fill_prefetch_buffer(chunk_size_frames * 2) # Maintain 2 chunks ahead

            # 7. Break condition if we've reached or passed the end_frame
            if chunk_end_frame >= end_frame:
                break

    def to_video(self) -> Video:
        """
        Returns a Video object from the video_framework for the entire video.
        """
        return Video(self.video_path, grayscale=self.grayscale)

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