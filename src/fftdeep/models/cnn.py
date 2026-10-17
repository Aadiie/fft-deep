# Building CNN 
from __future__ import annotations

import keras
from keras import layers


def build_cnn(
    filters=(32, 64, 128),
    convs_per_block: int = 1,
    batch_norm: bool = False,
    dropout: float = 0.3,
    augment: bool = True,
    input_shape=(32, 32, 3),
    name: str = "cnn",
) -> keras.Model:
    
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    if augment:
        x = layers.RandomFlip("horizontal")(x)  # only active during training

    for f in filters:
        for _ in range(convs_per_block):
            x = layers.Conv2D(f, 3, padding="same", use_bias=not batch_norm)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),  # as real =0 and fake =1 , FAKE = positive class
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model