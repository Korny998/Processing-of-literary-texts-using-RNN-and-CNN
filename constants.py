import os


# Root directory of the project
PROJECT_DIR: str = os.path.dirname(__file__)

# Setting for text proccesing
EMBEDDING_DIM: int = 300
FILTERS: str = (
    '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
)
MAX_WORDS: int = 15000

# Sequence window parameters
WIN_SIZE: int = 1000
WIN_STEP: int = 100

# Dataset balancing parameter
MIN_LEN_RATIO: float = 0.6