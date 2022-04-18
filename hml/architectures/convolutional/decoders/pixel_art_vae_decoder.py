import tensorflow as tf
from tensorflow.keras import layers


def model(latent_dim: int) -> tf.keras.Sequential:
    """
    VAE decoder
    """
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 1024, activation="relu")(latent_inputs)
    x = layers.Reshape((4, 4, 1024))(x)
    x = layers.Conv2DTranspose(
        512,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(x)
    x = layers.Conv2DTranspose(
        256,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(x)
    x = layers.Conv2DTranspose(
        128,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
    )(x)
    decoder_outputs = layers.Conv2DTranspose(
        3,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="sigmoid",
    )(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder
