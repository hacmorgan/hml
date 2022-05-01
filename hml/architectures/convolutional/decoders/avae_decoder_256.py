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
                4 * 4 * 1024,
                kernel_initializer=init,
            ),
            layers.Reshape((4, 4, 1024)),
            # Output shape: (4, 4, 1024)
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (8, 8, 512)
            layers.Conv2DTranspose(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (16, 16, 256)
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
                32,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (128, 128, 32)
            layers.Conv2DTranspose(
                16,
                kernel_size=5,
                strides=2,
                padding="same",
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape: (256, 256, 16)
            layers.Conv2DTranspose(
                3,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_initializer=init,
            ),
            # Output shape: (256, 256, 3)
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture
