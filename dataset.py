import os
import zipfile
from typing import Dict, List

import glob
import numpy as np
from tensorflow.keras import utils
from navec import Navec
from tensorflow.keras.preprocessing.text import Tokenizer

from constants import (
    EMBEDDING_DIM,
    FILTERS,
    MAX_WORDS,
    MIN_LEN_RATIO,
    PROJECT_DIR,
    WIN_SIZE, WIN_STEP
)


navec: Navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

data_path: str = utils.get_file(
    'russian_literature.zip',
    'https://storage.yandexcloud.net/academy.ai/russian_literature.zip'
)

data_dir: str = os.path.join(PROJECT_DIR, 'dataset')
os.makedirs(data_dir, exist_ok=True)

if not os.listdir(data_dir):
    with zipfile.ZipFile(data_path, 'r') as z:
        z.extractall(data_dir)

poems_dir: str = os.path.join(data_dir, 'poems')
prose_dir: str = os.path.join(data_dir, 'prose')

class_list: set[str] = set()

for folder in (poems_dir, prose_dir):
    if os.path.exists(folder):
        class_list.update(os.listdir(folder))

authors: List[str] = sorted(class_list)
all_texts: Dict[str, str] = {}

for author in authors:
    chunks: List[str] = []
    for path in (
        glob.glob(os.path.join(prose_dir, author, "*.txt"))
        + glob.glob(os.path.join(poems_dir, author, "*.txt"))
    ):
        with open(path, "r", errors="ignore") as file:
            chunks.append(file.read().replace("\n", " "))
    if chunks:
        all_texts[author] = " ".join(chunks)

authors = sorted(all_texts.keys())

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    filters=FILTERS,
    lower=True,
    split=' ',
    char_level=False
)

tokenizer.fit_on_texts(all_texts.values())
seq_train = tokenizer.texts_to_sequences([all_texts[a] for a in authors])

sizes = np.array([len(seq) for seq in seq_train])
median: int = int(np.median(sizes))

class_list_balance: List[str] = []
seq_train_balance = []

for author, seq in zip(authors, seq_train):
    if len(seq) > median * MIN_LEN_RATIO:
        class_list_balance.append(author)
        seq_train_balance.append(seq[:median])

sizes_balance = np.array([len(seq) for seq in seq_train_balance])


def seq_split(sequence, win_size: int, step: int) -> List:
    """Split a sequence into overlapping windows."""
    return [
        sequence[i:i + win_size] for i in range(
            0, len(sequence) - win_size + 1, step
        )
    ]


def seq_vectorize(
    seq_list: list,
    test_split: float,
    class_list: list,
    win_size: int,
    step: int
) -> tuple:
    """Convert sequences into train/test data for classification."""
    assert len(seq_list) == len(class_list)

    x_train, y_train, x_test, y_test = [], [], [], []
    num_classes = len(class_list)

    for cls in range(num_classes):
        sequence = seq_list[cls]
        gate = int(len(sequence) * (1 - test_split))

        train_windows = seq_split(sequence[:gate], win_size, step)
        test_windows = seq_split(sequence[gate:], win_size, step)

        if not train_windows:
            continue

        x_train.extend(train_windows)
        x_test.extend(test_windows)

        y_train.extend(
            [utils.to_categorical(cls, num_classes)] * len(train_windows)
        )
        y_test.extend(
            [utils.to_categorical(cls, num_classes)] * len(test_windows)
        )

    return (
        np.array(x_train, dtype=np.int32),
        np.array(y_train, dtype=np.float32),
        np.array(x_test, dtype=np.int32),
        np.array(y_test, dtype=np.float32)
    )


x_train, y_train, x_test, y_test = seq_vectorize(
    seq_train_balance,
    0.1,
    class_list_balance,
    WIN_SIZE,
    WIN_STEP
)


def load_embedding():
    """Build an embedding matrix aligned with the tokenizer vocabulary."""
    word_index = tokenizer.word_index
    embeddings_index = navec

    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM), dtype=np.float32)
    for word, i in word_index.items():
        if i < MAX_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix
