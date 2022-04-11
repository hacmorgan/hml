from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def model(
) -> tf.keras.Sequential:
    """
    A generic discriminator model
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential(
        [
            # Input size conv layer (e.g. 64, 64, 64)
            layers.Conv2D(
                64, kernel_size=5, strides=1, padding="same", input_shape=(64, 64, 3), kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Input size conv layer (e.g. 64, 64, 64)
            layers.Dropout(0.1),
            layers.BatchNormalization(),
            layers.Conv2D(
                64, kernel_size=5, strides=2, padding="same", kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Half input size conv layer (e.g. 32, 32, 128)
            layers.Dropout(0.1),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=5, strides=2, padding="same", kernel_initializer=init),
            layers.LeakyReLU(alpha=0.2),
            # Quarter input size conv layer (e.g. 16, 16, 256)
            layers.Dropout(0.1),
            layers.BatchNormalization(),
            layers.Conv2D(256, kernel_size=5, strides=2, padding="same", kernel_initializer=init),
            layers.LeakyReLU(alpha=0.2),
            # Output neuron
            layers.Dropout(0.1),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(1),
            layers.LeakyReLU(alpha=0.2),
        ]
    )

    model.summary()

    return model
