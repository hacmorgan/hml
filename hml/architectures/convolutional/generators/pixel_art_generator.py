import tensorflow as tf
from tensorflow.keras import layers


def model() -> tf.keras.Sequential:
    """
    A smaller version of the DGCAN paper model, to fit on a GTX980.
    """
    latent_dim = 100
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential(
        [
            # Latent input
            layers.Input(shape=(latent_dim,)),
            # Dense layer, shaped as (4, 4, 256) conv layer
            layers.BatchNormalization(),
            layers.Dense(4 * 4 * 256, use_bias=False, activation="relu", kernel_initializer=init),
            layers.Reshape((4, 4, 256)),
            # 1. upscale by fractionally strided conv, (8, 8, 256)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", use_bias=False, activation="relu", kernel_initializer=init,
            ),
            # 2. upscale by fractionally strided conv, (16, 16, 256)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", use_bias=False, activation="relu", kernel_initializer=init,
            ),
            # 3. upscale by fractionally strided conv, (32, 32, 128)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", use_bias=False, activation="relu", kernel_initializer=init,
            ),
            # 4. upscale by fractionally strided conv, (64, 64, 64)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                64, kernel_size=5, strides=2, padding="same", use_bias=False, activation="relu", kernel_initializer=init,
            ),
            # Output, (64, 64, 3)
            layers.Conv2D(3, kernel_size=7, padding="same", use_bias=False, activation="tanh", kernel_initializer=init)
        ]
    )
    model.summary()
    return model
