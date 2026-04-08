from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video_player import Video, Overlay, BoundingBox  # noqa: E402
from main import add_layers  # noqa: E402


def _write_tiny_video(path: Path, frames: int = 3, size: tuple[int, int] = (64, 64)) -> None:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    assert out.isOpened()
    w, h = size
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (10 + i * 5, 10 + i * 3), 4, (255, 255, 255), -1)
        out.write(frame)
    out.release()


def test_pipeline_can_process_first_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path)

    video = Video(str(video_path), grayscale=True)
    video = add_layers(video)
    frame, overlays = video.get_frame(0)

    assert frame.ndim == 3
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    assert isinstance(overlays, dict)


def test_overlay_rendering_smoke(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny2.mp4"
    _write_tiny_video(video_path)

    video = Video(str(video_path), grayscale=False)
    video.add_overlay_to_frame(
        0,
        Overlay(
            name="gt",
            overlay_items=[BoundingBox(5, 5, 12, 10, label="obj")],
        ),
    )
    frame, _ = video.get_frame(0)
    assert frame.ndim == 3
