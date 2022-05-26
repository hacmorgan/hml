from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def model(input_shape: Tuple[int, int, int] = (64, 64, 3)) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN paper's discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Input
            layers.InputLayer(input_shape=input_shape),
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
