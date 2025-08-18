from typing import List, Tuple, Optional, Deque, Callable, Dict, Any
from collections import deque
import cv2 as cv
import numpy as np
from video_streamer import Streamer
from video_player import JupyterPlayer, DesktopPlayer, Video, Point, Line, np_to_overlay_items, OverlayItem
from layers import (
    OpticalFlowLambda, 
    MotionStabilizer, 
    BackgroundSubtraction, 
    get_morphological_op, 
    HighPassFilter, 
    BandPassFilter, 
    MedianFilter, 
    CropImage,
    DetectClasses
)
from config import VideosConfig, OUTPUT_DIR, pjoin


def add_layers(video: Video) -> Video:
    flow_overlay = OpticalFlowLambda()
    motion_stabilizer = MotionStabilizer()
    detect_classes = DetectClasses()

    video.add_transform("Crop Image", CropImage(0.05))
    video.add_online_overlay(name="Optical Flow", overlay_func=flow_overlay)
    video.add_transform("Motion Stabilize", motion_stabilizer.get_stereo_warped_frame)
    video.add_transform("Median Blur", MedianFilter(3))
    video.add_transform("Band Pass Filter", BandPassFilter(0.3, 6))
    video.add_transform("Background Subtraction", BackgroundSubtraction("KNN"))
    video.add_transform("Morphological Operation", get_morphological_op(3, 4))
    #video.add_transform("Motion Stabilize Back", motion_stabilizer.warp_back)
    video.add_transform("Crop Image", CropImage(0.05))
    video.add_online_overlay(name="Detect Classes", overlay_func=detect_classes)

    return video


def get_video() -> Video:
    video = Video(VideosConfig.TRAIN_VIDEO, grayscale=True)
    video = add_layers(video)
    return video


def play() -> None:
    video = get_video()
    desktop_player = DesktopPlayer(video, output_dir=pjoin(OUTPUT_DIR, "debug"))
    desktop_player.show()


def save() -> None:
    video = get_video()
    video.save_video(output_path=pjoin(OUTPUT_DIR, "full_pipeline", "v2-no-tracking-final.mp4"),)


def main() -> None:
    save()


if __name__ == "__main__":
    main()
