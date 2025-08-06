import cv2
import numpy as np
from pathlib import Path
from video_player import Video, DesktopPlayer, load_bounding_boxes_from_csv, canny_edge_detector

def main():
    """A full example demonstrating the features of the video framework."""
    video_path = Path().resolve().parent.parent / "testing" / "resources" / "sample.mp4"
    annotations_path = Path().resolve().parent.parent / "testing" / "resources" / "annotations.csv"
    output_path = Path().resolve().parent.parent / "testing" / "output"

    with Video(str(video_path)) as video:
        # Load bounding boxes from the CSV file
        overlays = load_bounding_boxes_from_csv(str(annotations_path))
        video.add_overlays(overlays)

        # Add the Canny edge detector transformation
        video.add_transform(name="Canny Edges", transform_func=canny_edge_detector(threshold1=100, threshold2=200))
        video.add_transform(name="resize", transform_func=lambda frame: cv2.resize(frame, (960 // 2, 540 // 2)))

        # Define a predicate to save frames with a high number of edges
        def has_many_edges(frame: np.ndarray) -> bool:
            # A simple edge count check
            return bool(np.sum(frame > 0) > 500000) # Threshold for number of edge pixels

        # Save frames based on the predicate
        video.save_frames_where(has_many_edges, output_dir=str(output_path / "edge_frames"))

        # Play the video with interactive controls
        player = DesktopPlayer(video)
        player.show()

if __name__ == "__main__":
    main()