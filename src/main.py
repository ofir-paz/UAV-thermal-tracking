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
from sort import Sort
from config import VideosConfig, OUTPUT_DIR, pjoin


def add_layers(video: Video) -> Video:
    flow_overlay = OpticalFlowLambda(return_overlay_items=False)
    motion_stabilizer = MotionStabilizer(crop_percentage=0.05)
    detect_classes = DetectClasses(min_hits=30, max_age=20)

    video.add_online_overlay(name="Optical Flow", overlay_func=flow_overlay)
    video.add_transform("Motion Stabilize", motion_stabilizer.get_stereo_warped_frame)
    video.add_transform("Median Blur", MedianFilter(3))
    video.add_transform("Band Pass Filter", BandPassFilter(0.3, 6))
    video.add_transform("Background Subtraction", BackgroundSubtraction("KNN"))
    video.add_transform("Crop Image", motion_stabilizer.post_warp_crop)
    video.add_transform("Morphological Operation", get_morphological_op(3, 4))
    video.add_online_overlay(name="Detect Classes", overlay_func=detect_classes)
    #video.add_transform("Motion Stabilize Back", motion_stabilizer.warp_back)

    return video


def get_video() -> Video:
    video = Video(VideosConfig.TRAIN_VIDEO, grayscale=True)
    video = add_layers(video)
    return video


def play_desktop_player(video: Video) -> None:
    DesktopPlayer(video, output_dir=pjoin(OUTPUT_DIR, "debug")).show()


def play() -> None:
    video = get_video()
    play_desktop_player(video)


def play_remapped() -> None:
    video = get_video()
    video.set_play_mode('original_with_remapped')
    play_desktop_player(video)


def save() -> None:
    video = get_video()
    video.set_play_mode('original_with_remapped')
    video.save_video(output_path=pjoin(OUTPUT_DIR, "debug", "v2-tracking.mp4"),)


def main() -> None:
    play_remapped()


if __name__ == "__main__":
    main()