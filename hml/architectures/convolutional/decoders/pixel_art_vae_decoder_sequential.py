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
            # Dense layer, shaped as (4, 4, 1024) conv layer
            layers.BatchNormalization(),
            layers.Dense(
                4 * 4 * 1024,
                activation=tf.nn.relu,
                kernel_initializer=init,
            ),
            layers.Reshape((4, 4, 1024)),
            # 1. upscale by fractionally strided conv, (8, 8, 512)
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # 2. upscale by fractionally strided conv, (16, 16, 256)
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # 3. upscale by fractionally strided conv, (32, 32, 128)
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # 4. upscale by fractionally strided conv, (64, 64, 64)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # 4. upscale by fractionally strided conv, (128, 128, 32)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                32,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output layer
            layers.Conv2DTranspose(
                3,
                kernel_size=5,
                strides=1,
                padding="same",
                kernel_initializer=init,
            ),
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture
