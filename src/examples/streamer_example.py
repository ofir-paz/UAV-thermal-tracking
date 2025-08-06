from pathlib import Path
from video_streamer import Streamer

video_path = str(Path(__file__).resolve().parent.parent.parent / "testing" / "resources" / "sample.mp4")

print(f"Using video: {video_path}")

try:
    # Create a Streamer instance
    streamer = Streamer(video_path)
    print(f"Video FPS: {streamer.fps}")
    print(f"Video Frame Count: {streamer.frame_count}")
    print(f"Video Duration (seconds): {streamer.duration_seconds:.2f}")

    print("\n--- Streaming with chunk_size=5 seconds, overlap=0 seconds ---")
    # Stream with chunk_size of 5 seconds and no overlap
    for i, (start_frame, end_frame, frames) in enumerate(streamer.stream(chunk_size=5, overlap=0, unit='seconds')):
        print(f"Chunk {i+1}: Start Frame={start_frame}, End Frame={end_frame}, Frames in Chunk={len(frames)}")

    print("\n--- Streaming with chunk_size=200 frames, overlap=50 frames ---")
    # Stream with chunk_size of 200 frames and 50 frames overlap
    for i, (start_frame, end_frame, frames) in enumerate(streamer.stream(chunk_size=200, overlap=50, unit='frames')):
        print(f"Chunk {i+1}: Start Frame={start_frame}, End Frame={end_frame}, Frames in Chunk={len(frames)}")

    print("\n--- Streaming from 10 to 20 seconds with chunk_size=3 seconds ---")
    # Stream a specific segment from 10 to 20 seconds with 3-second chunks
    for i, chunk in enumerate(streamer.stream(start=10, end=20, chunk_size=3, unit='seconds')):
        print(f"Chunk {i+1}: Start Frame={chunk.start}, End Frame={chunk.end}, Frames in Chunk={len(chunk.frames)}")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the video file exists.")
except ValueError as e:
    print(f"Error: {e}. Please check the video file or streamer parameters.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
