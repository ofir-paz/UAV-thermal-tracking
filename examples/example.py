
from video_framework import Video, Player

def main():
    """A simple example of how to use the video framework."""
    video_path = "testing/resources/sample.mp4"
    
    with Video(video_path) as video:
        player = Player(video)
        player.play()

if __name__ == "__main__":
    main()
