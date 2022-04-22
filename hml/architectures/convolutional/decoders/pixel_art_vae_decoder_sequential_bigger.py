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
                8 * 8 * 1024,
                activation=tf.nn.relu,
                kernel_initializer=init,
            ),
            layers.Reshape((8, 8, 1024)),
            # Output shape: (8, 8, 1024)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output shape: (16, 16, 512)
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layers.Conv2DTranspose(
            #     512,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # Output shape: (16, 16, 512)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output shape: (32, 32, 256)
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layers.Conv2DTranspose(
            #     256,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # Output shape: (32, 32, 256)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output shape: (64, 64, 128)
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layers.Conv2DTranspose(
            #     128,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # Output shape: (64, 64, 128)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output shape: (128, 128, 64)
            # layers.Dropout(0.2),
            # layers.BatchNormalization(),
            # layers.Conv2DTranspose(
            #     64,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # Output shape: (128, 128, 64)
            layers.BatchNormalization(),
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
