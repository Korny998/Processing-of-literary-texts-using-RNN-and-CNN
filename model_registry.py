from models import (
    build_Conv1D, build_Conv_LSTM,
    build_GRU_1, build_GRU_2,
    build_LSTM_1, build_LSTM_2,
    build_MIX_1, build_simple_rnn_1, build_simple_rnn_2
)


MODEL_REGISTRY = [
    ('simple_rnn_1', build_simple_rnn_1),
    ('simple_rnn_2', build_simple_rnn_2),
    ('gru_1', build_GRU_1),
    ('gru_2', build_GRU_2),
    ('lstm_1', build_LSTM_1),
    ('lstm_2', build_LSTM_2),
    ('mix_1', build_MIX_1),
    ('conv1d', build_Conv1D),
    ('conv_lstm', build_Conv_LSTM)
]
