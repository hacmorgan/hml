from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def model(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> tf.keras.Sequential:
    """
    A simple discriminator model
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            # Input: (256, 256, 3)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (128, 128, 128)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (64, 64, 128)
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
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (16, 16, 128)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (8, 8, 128)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (4, 4, 128)
            layers.Flatten(),
            layers.Dense(1),
            layers.Activation("sigmoid"),
            # Discriminator output
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
