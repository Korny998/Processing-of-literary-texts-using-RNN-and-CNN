from dataclasses import dataclass
from typing import Tuple

from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    batch_size: int
    epochs: int
    freeze_embedding: bool = True


class Trainer:
    """A reusable trainer for Keras classification models."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def _freeze_embedding(self, model: Model):
        """Freeze the first Embedding layer."""
        if not self.config.freeze_embedding:
            return

        for layer in model.layers:
            if isinstance(layer, Embedding):
                layer.trainable = False
                break

    def train(
        self, build_fn,
        x_train, y_train,
        x_val, y_val
    ) -> Tuple[Model, History]:
        """Build, compile and train a model."""
        model = build_fn()
        self._freeze_embedding(model)

        model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(x_val, y_val)
        )

        return model, history
