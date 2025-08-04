import cv2
from video_framework import Video, Player, to_grayscale

def main():
    """A simple example of how to use the video framework with transformations."""
    video_path = "testing/resources/sample.mp4"
    
    with Video(video_path) as video:
        # Add the grayscale transformation
        video.add_transform(to_grayscale)
        def add_threshold(frame):
            """A simple thresholding transformation."""
            _, thresh = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)
            return thresh
        video.add_transform(add_threshold)

        player = Player(video)
        player.play()

if __name__ == "__main__":
    main()
