from keras import models, layers

from constants import (
    EMBEDDING_DIM, MAX_WORDS, WIN_SIZE
)
from dataset import class_list_balance, load_embedding


def build_simple_rnn_1():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.3),
        layers.BatchNormalization(),
        layers.SimpleRNN(10),
        layers.Dropout(0.3),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_simple_rnn_2():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.BatchNormalization(),
        layers.SimpleRNN(5),
        layers.Dropout(0.2),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_GRU_1():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.BatchNormalization(),
        layers.GRU(
            10,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu'
        ),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_GRU_2():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.BatchNormalization(),
        layers.GRU(
            40,
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='relu'
        ),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_LSTM_1():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.BatchNormalization(),
        layers.LSTM(20),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_LSTM_2():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.3),
        layers.BatchNormalization(),
        layers.LSTM(100),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_MIX_1():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.3),
        layers.BatchNormalization(),
        layers.Bidirectional(
            layers.LSTM(8, return_sequences=True)
        ),
        layers.Bidirectional(
            layers.LSTM(8, return_sequences=True)
        ),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.GRU(
            16,
            return_sequences=True,
            reset_after=True
        ),
        layers.GRU(
            16,
            reset_after=True
        ),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(
            100,
            activation='relu'
        ),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_Conv1D():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.BatchNormalization(),
        layers.Conv1D(
            20, 5,
            activation='relu',
            padding='same'
        ),
        layers.Conv1D(
            20, 5,
            activation='relu'
        ),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])


def build_Conv_LSTM():
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(1, return_sequences=1),
        layers.Dense(100, activation='relu'),
        layers.Conv1D(
            20, 5,
            activation='relu'
        ),
        layers.LSTM(4, return_sequences=1),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv1D(
            20, 5,
            activation='relu'
        ),
        layers.Conv1D(
            20, 5,
            activation='relu'
        ),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Flatten()
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])
