"""Two model options for binary image classification (e.g. "finding" vs. "no finding").

- small_cnn: trains from scratch on CPU in seconds; used in the tests.
- resnet50_transfer: ImageNet-pretrained ResNet50 backbone (weights are
  downloaded by Keras on first use) with a new classification head.

Research/education code: not a medical device, not for clinical use.
"""

from __future__ import annotations

import keras
from keras import layers


def augmentation() -> keras.Sequential:
    # Mild geometric/contrast changes. No horizontal flip: anatomy is not symmetric.
    return keras.Sequential([layers.RandomRotation(0.03), layers.RandomZoom(0.1), layers.RandomContrast(0.1)],
                            name="augmentation")


def small_cnn(size: int = 128) -> keras.Model:
    inputs = keras.Input((size, size, 1))
    x = augmentation()(inputs)
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalMaxPooling2D()(x)  # a small local finding should count, wherever it is
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, outputs, name="small_cnn"))


def resnet50_transfer(size: int = 224, weights: str | None = "imagenet", train_backbone: bool = False) -> keras.Model:
    inputs = keras.Input((size, size, 1))
    x = augmentation()(inputs)
    x = layers.Concatenate()([x, x, x])  # grayscale -> 3 channels for the ImageNet backbone
    x = layers.Rescaling(255.0)(x)  # inputs are in [0, 1]; ResNet preprocessing expects [0, 255]
    x = keras.applications.resnet.preprocess_input(x)
    backbone = keras.applications.ResNet50(include_top=False, weights=weights, input_shape=(size, size, 3))
    backbone.trainable = train_backbone  # start with a frozen backbone, fine-tune later
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, outputs, name="resnet50_transfer"))


def _compile(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="sensitivity")])
    return model
