
import cv2
from video_framework import Video, Player, Overlay, BoundingBox, to_grayscale


def main():
    """A simple example of how to use the video framework with overlays."""
    video_path = "testing/resources/sample.mp4"
    
    with Video(video_path) as video:
        # Create a simple bounding box
        bbox = BoundingBox(x=100, y=100, width=200, height=150, label="Test Box")
        
        # Create an overlay with the bounding box
        overlay = Overlay(bounding_boxes=[bbox])
        
        # Add the overlay to the video
        overlays = {i: overlay for i in range(video.frame_count)}
        video.add_overlays(overlays)
        def add_threshold(frame):
            """A simple thresholding transformation."""
            _, thresh = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
            return thresh
        video.add_transform(to_grayscale)
        video.add_transform(add_threshold)

        player = Player(video)
        player.play()

if __name__ == "__main__":
    main()
