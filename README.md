# UAV Thermal Tracking

Thermal video processing and object tracking pipeline for UAV footage.

## Repository layout

- `src/main.py` – main pipeline and entrypoint
- `src/layers.py` – processing layers (stabilization, filtering, detection, tracking)
- `src/video_player` – internal video player/overlay framework (internalized mini-repo)
- `src/video_streamer.py` – chunk-based video streaming utility
- `testing/resources` – sample assets used by examples/tests

## Python version

Use Python `3.11` (pinned in `.python-version`).

## Setup

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Run

Validated single entrypoint:

```bash
python src/main.py
```

## Pipeline summary

The main processing chain in `src/main.py` is:

1. Optical flow overlay (state initialization)
2. Motion stabilization
3. Temporal median filter
4. Band-pass filtering
5. Background subtraction (KNN)
6. Post-warp crop
7. Morphological cleanup
8. Class detection overlay
9. Object tracking overlay
10. Legend overlay

## Notes on tracking implementation

`src/sort.py` now contains a project-owned, MIT-compatible SORT-style tracker implementation used when `library="SORT"`.

If you select `library="Trackers"`, install and pin the external `trackers` package in your environment explicitly.

## Known limitations

- Tracking quality and thresholds are tuned for project data; additional domains require retuning.
- Real-time performance depends on CPU and video resolution.
- Jupyter/desktop players are utility viewers, not production serving components.
