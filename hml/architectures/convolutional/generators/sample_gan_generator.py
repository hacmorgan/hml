import tensorflow as tf
from tensorflow.keras import layers


GENERATOR_LATENT_DIM = 128


def model() -> tf.keras.Sequential:
    """
    SampleGAN model: generates 1920x1080 px wallpaper
    """
    generator = tf.keras.Sequential(
        [
            # Latent input
            layers.Input(shape=(GENERATOR_LATENT_DIM,)),
            # Dense layer, shaped as (16, 9, 128) conv layer
            layers.Dense(9 * 16 * 128, use_bias=False),
            layers.Reshape((9, 16, 128)),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (9, 16, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (18, 32, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (36, 64, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (72, 128, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (144, 256, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (288, 512, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (576, 1024, 128)
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Output shape (1152, 2048, 128)
            layers.Conv2D(
                3,
                kernel_size=7,
                padding="same",
                activation="sigmoid",
                use_bias=False,
            ),
        ]
    )
    generator.summary()
    return generator
