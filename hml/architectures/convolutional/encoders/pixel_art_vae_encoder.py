import tensorflow as tf
from tensorflow.keras import layers

from hml.layers.VAESampling import VAESampling


def model(latent_dim: int) -> tf.keras.Sequential:
    """
    VAE decoder
    """
    encoder_inputs = layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(
        128,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(encoder_inputs)
    x = layers.Conv2D(
        256,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(x)
    x = layers.Conv2D(
        512,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(x)
    x = layers.Conv2D(
        1024,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="sigmoid",
    )(x)
    x = layers.Dense(100, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = VAESampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder
