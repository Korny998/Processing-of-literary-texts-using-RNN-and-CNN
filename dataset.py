import os
import zipfile

import glob
import numpy as np
from tensorflow.keras import utils
from navec import Navec
from tensorflow.keras.preprocessing.text import Tokenizer

from constants import (
    CLASS_LIST, EMBEDDING_DIM, FILTERS,
    MAX_WORDS, MAX_SEQ, MIN_LEN_RATIO,
    PROJECT_DIR, WIN_SIZE, WIN_STEP
)


navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

data_path = utils.get_file(
    'russian_literature.zip',
    'https://storage.yandexcloud.net/academy.ai/russian_literature.zip'
)

data_dir: str = os.path.join(PROJECT_DIR, 'dataset')
os.makedirs(data_dir, exist_ok=True)

if not os.listdir(data_dir):
    with zipfile.ZipFile(data_path, 'r') as z:
        z.extractall(data_dir)

class_list = set()

for path in ('./dataset/poems', './dataset/prose'):
    CLASS_LIST.update(os.listdir(path))

all_texts: dict = {}

for author in sorted(class_list):
    all_texts[author] = ''
    for path in (
        glob.glob(os.path.join(data_dir, 'prose', author, '*.txt')) +
        glob.glob(os.path.join(data_dir, 'poems', author, '*.txt'))
    ):
        with open(path, 'r', errors='ignore') as file:
            text = file.read()
            all_texts[author] += text.replace('\n', ' ')

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    filters=FILTERS,
    lower=True,
    split=' ',
    char_level=False
)

tokenizer.fit_on_texts(all_texts.values())
seq_train = tokenizer.texts_to_sequences(all_texts.avlues())

sizes = np.array([len(seq) for seq in seq_train])
median = int(np.median(sizes))

class_list_balance = []
seq_train_balance = []

for author, seq in zip(class_list, seq_train):
    if len(seq) > median * MIN_LEN_RATIO:
        class_list_balance.append(author)
        seq_train_balance.append(seq[:median])

sizes_balance = np.array([len(seq) for seq in seq_train_balance])


def seq_split(sequence, win_size: int, step: int) -> list:
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


x_x_train, y_train, x_test, y_test = seq_vectorize(
    seq_train_balance,
    0.1,
    class_list_balance,
    WIN_SIZE,
    WIN_STEP
)
