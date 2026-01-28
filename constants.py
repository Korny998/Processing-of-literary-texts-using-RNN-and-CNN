import os


# Root directory of the project
PROJECT_DIR: str = os.path.dirname(__file__)

# Text processing / embedding settings
EMBEDDING_DIM: int = 300
FILTERS: str = (
    '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
)
MAX_WORDS: int = 15000

# Sequence window parameters (sliding window over token indices)
WIN_SIZE: int = 1000
WIN_STEP: int = 100

# Dataset balancing parameter
MIN_LEN_RATIO: float = 0.6

# Training hyperparameters
BATCH_SIZE: int = 128
EPOCHS: int = 10

# Whether to freeze the embedding layer during training
FREEZE_EMBEDDING: bool = True
