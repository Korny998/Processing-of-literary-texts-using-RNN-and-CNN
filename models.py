from tensorflow.keras import models, layers
from tensorflow.keras.models import Model

from constants import (
    EMBEDDING_DIM, MAX_WORDS, WIN_SIZE
)
from dataset import class_list_balance, load_embedding


def build_simple_rnn_1() -> Model:
    """SimpleRNN baseline with moderate dropout and batch normalization."""
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


def build_simple_rnn_2() -> Model:
    """Smaller SimpleRNN variant with lighter dropout."""
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


def build_GRU_1() -> Model:
    """GRU model with recurrent dropout and ReLU activation."""
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


def build_GRU_2() -> Model:
    """Larger GRU variant (more units) for potentially higher capacity."""
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


def build_LSTM_1() -> Model:
    """LSTM model with batch normalization."""
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


def build_LSTM_2() -> Model:
    """High-capacity LSTM variant with stronger dropout."""
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


def build_MIX_1() -> Model:
    """Mixed architecture: BiLSTM blocks + GRU blocks + dense head."""
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


def build_Conv1D() -> Model:
    """1D CNN architecture for local n-gram-like feature extraction."""
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


def build_Conv_LSTM() -> Model:
    """Hybrid model combining LSTM and Conv1D blocks."""
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS,
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[load_embedding()]
        ),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(1, return_sequences=True),
        layers.Dense(100, activation='relu'),
        layers.Conv1D(
            20, 5,
            activation='relu'
        ),
        layers.LSTM(4, return_sequences=True),
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
        layers.Flatten(),
        layers.Dense(
            len(class_list_balance),
            activation='softmax'
        )
    ])
