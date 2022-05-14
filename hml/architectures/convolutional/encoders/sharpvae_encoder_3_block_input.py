from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def model(
    latent_dim: int, input_shape: Tuple[int, int, int] = (192, 192, 3)
) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN paper's discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Input
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(
                64,
                kernel_size=5,
                strides=3,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (64, 64, 64)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (32, 32, 128)
            layers.Conv2D(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (16, 16, 256)
            layers.Conv2D(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (8, 8, 512)
            layers.Conv2D(
                1024,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (4, 4, 1024)
            # layers.Conv2D(
            #     1024,
            #     kernel_size=5,
            #     strides=2,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # # Output shape: (2, 2, 1024)
            # layers.Conv2D(
            #     1024,
            #     kernel_size=5,
            #     strides=2,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            # # Output shape: (1, 1, 1024)
            layers.Flatten(),
            layers.Dense(2 * latent_dim, kernel_initializer=init),
            # Latent output (No activation)
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
