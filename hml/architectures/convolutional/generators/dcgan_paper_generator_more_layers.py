import tensorflow as tf
from tensorflow.keras import layers


def model() -> tf.keras.Sequential:
    """
    A smaller version of the DGCAN paper model, to fit on a GTX980.
    """
    latent_dim = 100
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Latent input
            layers.Input(shape=(latent_dim,)),
            # Dense layer, shaped as (4, 4, 1024) conv layer
            layers.BatchNormalization(),
            layers.Dense(
                4 * 4 * 1024, use_bias=False, activation="relu", kernel_initializer=init
            ),
            layers.Reshape((4, 4, 1024)),
            # 1.a. upscale by fractionally strided conv, (8, 8, 512)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 1.b. plain conv, (8, 8, 512)
            layers.BatchNormalization(),
            layers.Conv2D(
                512,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 2.a. upscale by fractionally strided conv, (16, 16, 256)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 2.b. plain conv, (16, 16, 256)
            layers.BatchNormalization(),
            layers.Conv2D(
                256,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 3.a. upscale by fractionally strided conv, (32, 32, 128)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 3.b. plain conv, (32, 32, 128)
            layers.BatchNormalization(),
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 4.a upscale by fractionally strided conv, (64, 64, 3)
            layers.Conv2DTranspose(
                3,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="relu",
                kernel_initializer=init,
            ),
            # 4.b Output conv, (64, 64, 3)
            layers.Conv2D(
                3,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                activation="tanh",
                kernel_initializer=init,
            ),
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture
