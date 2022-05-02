import tensorflow as tf
from tensorflow.keras import layers


def model(latent_dim: int) -> tf.keras.Sequential:
    """
    A relatively simple encoder
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Input
            layers.InputLayer(input_shape=(256, 256, 3)),
            layers.Conv2D(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (128, 128, 64)
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
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (32, 32, 256)
            layers.Conv2D(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (16, 16, 512)
            layers.Conv2D(
                1024,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (8, 8, 1024)
            layers.Conv2D(
                2048,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (4, 4, 2048)
            layers.Conv2D(
                4096,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output shape: (2, 2, 4096)
            layers.Flatten(),
            layers.Dense(2 * latent_dim, kernel_initializer=init),
            # Latent output (No activation)
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
