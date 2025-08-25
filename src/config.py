from os.path import join as pjoin
from pathlib import Path
from typing import final
from collections import OrderedDict
from enum import IntEnum
import numpy as np
from video_player import Color


# Set settings
np.set_printoptions(precision=5, suppress=True, linewidth=120)


# Paths
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = pjoin(ROOT_DIR, "data")
TESTING_DIR = pjoin(ROOT_DIR, "testing")
OUTPUT_DIR = pjoin(TESTING_DIR, "output")


@final
class VideosConfig:
    """
    Configuration class for video paths.
    """
    INPP = pjoin(DATA_DIR, "inpp.mp4")
    MOVE1 = pjoin(DATA_DIR, "move1.mp4")
    MOVING_CHECK = pjoin(DATA_DIR, "moving_check.mp4")
    MV1 = pjoin(DATA_DIR, "mv1.mp4")
    PERSONS = pjoin(DATA_DIR, "persons.mp4")
    FPS = FRAME_RATE = 29.994
    
    VIDEOS = OrderedDict(
        inpp=INPP,
        move1=MOVE1,
        moving_check=MOVING_CHECK,
        mv1=MV1,
        persons=PERSONS
    )

    EXAMPLE_VIDEO = MOVING_CHECK
    TRAIN_VIDEO = MOVING_CHECK
    
    TEST_VIDEOS = OrderedDict(
        inpp=INPP,
        move1=MOVE1,
        mv1=MV1,
        persons=PERSONS
    )


class Classes(IntEnum):
    COLOR: Color
    TEXT: str
    
    def __new__(cls, value, color, text):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.COLOR = Color(color)
        obj.TEXT = text
        return obj

    BACKGROUND = 0, Color("#cdcdcd"), "Background"
    PERSON     = 1, Color("#0000ff"), "Person"
    VEHICLE    = 2, Color("#00ff00"), "Vehicle"
