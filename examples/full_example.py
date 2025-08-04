import cv2
from video_framework import Video, Player, load_bounding_boxes_from_csv, canny_edge_detector
import numpy as np

def main():
    """A full example demonstrating the features of the video framework."""
    video_path = "testing/resources/sample.mp4"
    annotations_path = "testing/resources/annotations.csv"

    with Video(video_path) as video:
        # Load bounding boxes from the CSV file
        overlays = load_bounding_boxes_from_csv(annotations_path)
        video.add_overlays(overlays)

        # Add the Canny edge detector transformation
        video.add_transform(name="Canny Edges", transform_func=canny_edge_detector(threshold1=100, threshold2=200))
        video.add_transform(name="resize", transform_func=lambda frame: cv2.resize(frame, (960 // 2, 540 // 2)))

        # Define a predicate to save frames with a high number of edges
        def has_many_edges(frame: np.ndarray) -> bool:
            # A simple edge count check
            return bool(np.sum(frame > 0) > 100000) # Threshold for number of edge pixels

        # Save frames based on the predicate
        video.save_frames_where(has_many_edges, output_dir="examples/edge_frames")

        # Play the video with interactive controls
        player = Player(video)
        player.play()

if __name__ == "__main__":
    main()