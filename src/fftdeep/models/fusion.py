#three CNN models for  Ai image detection 
# one using original images, one using FFT spectra, and one combining both.
from __future__ import annotations

import keras
from keras import layers


def conv_stack(x, filters=(32, 64, 128), convs_per_block=2):
    for f in filters:
        for _ in range(convs_per_block):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    return layers.GlobalAveragePooling2D()(x)


def build_model(mode: str, dropout: float = 0.3) -> keras.Model:
    """mode: 'image', 'freq' or 'both'. Output: P(FAKE)."""
    if mode not in {"image", "freq", "both"}:
        raise ValueError(f"unknown mode: {mode}")
    inputs, feats = {}, []
    if mode in ("image", "both"):
        img = keras.Input(shape=(32, 32, 3), name="image")          # float 0-255
        inputs["image"] = img
        feats.append(conv_stack(layers.Rescaling(1.0 / 255)(img)))
    if mode in ("freq", "both"):
        spec = keras.Input(shape=(32, 32, 1), name="spectrum")      # standardised spectrum
        inputs["spectrum"] = spec
        feats.append(conv_stack(spec))
    x = feats[0] if len(feats) == 1 else layers.Concatenate()(feats)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out, name=f"cnn_{mode}")