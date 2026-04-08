"""Entry point for the final UAV thermal tracking pipeline."""

from typing import Callable, Sequence, Tuple

from video_player import DesktopPlayer, Video
from layers import (
    OpticalFlowLambda,
    MotionStabilizer,
    BackgroundSubtraction,
    get_morphological_op,
    BandPassFilter,
    MedianFilter, 
    DetectClasses,
    TrackDetectedObjects,
    legend_overlay
)
from config import VideosConfig, OUTPUT_DIR, pjoin


TransformStep = Tuple[str, Callable]
OverlayStep = Tuple[str, Callable]


def _build_pipeline_components() -> Tuple[MotionStabilizer, OpticalFlowLambda, DetectClasses, TrackDetectedObjects]:
    """Create the stateful algorithm components used by the processing pipeline."""
    flow_overlay = OpticalFlowLambda(return_overlay_items=False)
    motion_stabilizer = MotionStabilizer(crop_percentage=0.05, fixer_ema_factor=0.975)
    detect_classes = DetectClasses(dilate_size=0, return_overlay_items=False)
    tracker = TrackDetectedObjects(max_age=35, min_hits=35, iou_threshold=0.1, score_threshold=0.47, library="Trackers")
    return motion_stabilizer, flow_overlay, detect_classes, tracker


def _transform_steps(motion_stabilizer: MotionStabilizer) -> Sequence[TransformStep]:
    """Ordered frame-processing stages of the final algorithm."""
    return (
        ("Motion Stabilize", motion_stabilizer.get_corrected_frame),
        ("Temporal Median", MedianFilter(1, 5, 2)),
        ("Band Pass Filter", BandPassFilter(0.5, 6)),
        ("Background Subtraction", BackgroundSubtraction("KNN")),
        ("Crop Image", motion_stabilizer.post_warp_crop),
        ("Morphological Operation", get_morphological_op(3, 4, (7, 9))),
    )


def _overlay_steps(flow_overlay: OpticalFlowLambda, detect_classes: DetectClasses, tracker: TrackDetectedObjects) -> Sequence[OverlayStep]:
    """Ordered online overlays for visualization and tracking output."""
    return (
        ("Optical Flow", flow_overlay),
        ("Detect Classes", detect_classes),
        ("Track Detected Objects", tracker),
        ("legend", legend_overlay),
    )


def add_layers(video: Video) -> Video:
    motion_stabilizer, flow_overlay, detect_classes, tracker = _build_pipeline_components()

    for name, transform in _transform_steps(motion_stabilizer):
        video.add_transform(name, transform)

    for name, overlay in _overlay_steps(flow_overlay, detect_classes, tracker):
        video.add_online_overlay(name=name, overlay_func=overlay)

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
    video.save_video(output_path=pjoin(OUTPUT_DIR, "debug", "v2.3-tracking.mp4"),)


def main() -> None:
    play_remapped()


if __name__ == "__main__":
    main()
