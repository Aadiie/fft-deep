#Grad Cam
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def load_logit_model(path):
    """Load a saved model; swap the final sigmoid for a linear output (the logit)."""
    model = keras.models.load_model(path, compile=False)
    model.layers[-1].activation = keras.activations.linear
    return model


def grad_cam(logit_model, images):
    """images: float32 (N, 32, 32, 3), values 0-255.
    Returns heatmaps (N, 32, 32) in [0, 1] and logits (N,)."""
    last_relu = [l for l in logit_model.layers if isinstance(l, keras.layers.ReLU)][-1]
    grad_model = keras.Model(logit_model.inputs, [last_relu.output, logit_model.output])
    x = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        maps, logit = grad_model(x, training=False)      # maps: (N, 8, 8, 128)
        logit = logit[:, 0]
        target = tf.where(logit > 0, logit, -logit)      # evidence for the predicted class
    grads = tape.gradient(target, maps)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * maps, axis=-1))
    cam = tf.image.resize(cam[..., None], (32, 32))[..., 0].numpy()
    cam = cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-8)
    return cam, logit.numpy()


def drop_test(logit_model, images, cams, frac=0.25, seed=0):
    """Hide the top-Grad-CAM pixels vs the same number of random pixels.
    Returns (drop_top, drop_random): average fall in evidence for the predicted class."""
    rng = np.random.default_rng(seed)
    base = logit_model.predict(images, verbose=0).ravel()
    sign = np.where(base > 0, 1.0, -1.0)
    k = int(frac * 32 * 32)
    mean = images.mean(axis=(1, 2), keepdims=True)
    top, rnd = images.copy(), images.copy()
    for n in range(len(images)):
        order = np.argsort(cams[n].ravel())[::-1][:k]
        r, c = np.unravel_index(order, (32, 32))
        top[n, r, c] = mean[n, 0, 0]
        r, c = np.unravel_index(rng.choice(32 * 32, k, replace=False), (32, 32))
        rnd[n, r, c] = mean[n, 0, 0]
    def drop(x):
        return float((sign * (base - logit_model.predict(x, verbose=0).ravel())).mean())
    return drop(top), drop(rnd)