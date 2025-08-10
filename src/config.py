from os.path import join as pjoin
from pathlib import Path
from typing import final
from collections import OrderedDict


# Paths
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = pjoin(ROOT_DIR, "data")


@final
class VideosConfig:
    """
    Configuration class for video paths.
    """
    INPP = pjoin(DATA_DIR, "inpp")
    MOVE1 = pjoin(DATA_DIR, "move1.mp4")
    MOVING_CHECK = pjoin(DATA_DIR, "moving_check.mp4")
    MV1 = pjoin(DATA_DIR, "mv1.mp4")
    PERSONS = pjoin(DATA_DIR, "persons.mp4")
    
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
