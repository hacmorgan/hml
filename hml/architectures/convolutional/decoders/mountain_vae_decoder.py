import tensorflow as tf
from tensorflow.keras import layers


def model(latent_dim: int) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Latent input
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(
                4 * 4 * 128,
                kernel_initializer=init,
            ),
            layers.Reshape((4, 4, 128)),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (4, 4, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (8, 8, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (16, 16, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (32, 32, 128)
            layers.Conv2DTranspose(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (64, 64, 64)
            layers.Conv2DTranspose(
                3,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_initializer=init,
            ),
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture