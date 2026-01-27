from keras.losses import CategoricalCrossentropy  
from keras.optimizers import Adam

from constants import BATCH_SIZE, EPOCHS
from dataset import (
    class_list_balance, x_train, y_train, x_test, y_test
)
from graphs_example import (
    show_confusion_matrix, show_plot
)
from models import (
    build_Conv1D, build_Conv_LSTM,
    build_GRU_1, build_GRU_2,
    build_LSTM_1, build_LSTM_2,
    build_MIX_1,
    build_simple_rnn_1, build_simple_rnn_2
)


def train_simple_rnn_1():
    model = build_simple_rnn_1()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_simple_rnn_2():
    model = build_simple_rnn_2()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_GRU_1():
    model = build_GRU_1()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_GRU_2():
    model = build_GRU_2()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_LSTM_1():
    model = build_LSTM_1()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_LSTM_2():
    model = build_LSTM_2()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_MIX_1():
    model = build_MIX_1()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_Conv1D():
    model = build_Conv1D()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


def train_Conv_LSTM():
    model = build_Conv_LSTM()
    model.layers[0].trainable = False
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test)
    )
    return model, history


if __name__ == '__main__':
    model_simple_rnn_1, history_simple_rnn_1 = train_simple_rnn_1()
    y_pred_simple_rnn_1 = model_simple_rnn_1.predict(x_test)
    show_plot(history_simple_rnn_1)
    show_confusion_matrix(y_test, y_pred_simple_rnn_1, class_list_balance)
    
    model_simple_rnn_2, history_simple_rnn_2 = train_simple_rnn_2()
    y_pred_simple_rnn_2 = model_simple_rnn_2.predict(x_test)
    show_plot(history_simple_rnn_2)
    show_confusion_matrix(y_test, y_pred_simple_rnn_2, class_list_balance)
    
    model_GRU_1, history_GRU_1 = train_GRU_1()
    y_pred_GRU_1 = model_GRU_1.predict(x_test)
    show_plot(history_GRU_1)
    show_confusion_matrix(y_test, y_pred_GRU_1, class_list_balance)
    
    model_GRU_2, history_GRU_2 = train_GRU_2()
    y_pred_GRU_2 = model_GRU_2.predict(x_test)
    show_plot(history_GRU_2)
    show_confusion_matrix(y_test, y_pred_GRU_2, class_list_balance)
    
    model_LSTM_1, history_LSTM_1 = train_LSTM_1()
    y_pred_LSTM_1 = model_LSTM_1.predict(x_test)
    show_plot(history_LSTM_1)
    show_confusion_matrix(y_test, y_pred_LSTM_1, class_list_balance)
    
    model_LSTM_2, history_LSTM_2 = train_LSTM_2()
    y_pred_LSTM_2 = model_LSTM_2.predict(x_test)
    show_plot(history_LSTM_2)
    show_confusion_matrix(y_test, y_pred_LSTM_2, class_list_balance)
    
    model_MIX_1, history_MIX_1 = train_MIX_1()
    y_pred_MIX_1 = model_MIX_1.predict(x_test)
    show_plot(history_MIX_1)
    show_confusion_matrix(y_test, y_pred_MIX_1, class_list_balance)
    
    model_Conv1D, history_Conv1D = train_Conv1D()
    y_pred_Conv1D = model_Conv1D.predict(x_test)
    show_plot(history_Conv1D)
    show_confusion_matrix(y_test, y_pred_Conv1D, class_list_balance)
    
    model_Conv_LSTM, history_Conv_LSTM = train_Conv_LSTM()
    y_pred_Conv_LSTM = model_Conv_LSTM.predict(x_test)
    show_plot(history_Conv_LSTM)
    show_confusion_matrix(y_test, y_pred_Conv_LSTM, class_list_balance)
