from tensorflow.keras import backend

from constants import BATCH_SIZE, EPOCHS, FREEZE_EMBEDDING
from dataset import class_list_balance, x_train, y_train, x_test, y_test
from graphs_example import show_confusion_matrix, show_plot
from model_registry import MODEL_REGISTRY
from trainer import Trainer, TrainerConfig


def main() -> None:
    """Train all registered models and visualize their performance."""
    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        freeze_embedding=FREEZE_EMBEDDING
    )

    trainer = Trainer(config)

    for name, build_fn in MODEL_REGISTRY:
        print(f'Training model: {name}')

        model, history = trainer.train(
            build_fn=build_fn,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test
        )

        y_pred = model.predict(x_test)

        show_plot(history)
        show_confusion_matrix(y_test, y_pred, class_list_balance)

        backend.clear_session()


if __name__ == '__main__':
    main()
