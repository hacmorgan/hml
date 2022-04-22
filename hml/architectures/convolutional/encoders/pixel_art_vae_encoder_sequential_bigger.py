import tensorflow as tf
from tensorflow.keras import layers


def model(latent_dim: int) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN paper's discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Input
            layers.InputLayer(input_shape=(128, 128, 3)),
            layers.Conv2D(
                64,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Output shape: (128, 128, 64)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2D(
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
            # layers.Conv2D(
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
            layers.Conv2D(
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
            # layers.Conv2D(
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
            layers.Conv2D(
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
            # layers.Conv2D(
            #     512,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     activation="relu",
            #     kernel_initializer=init,
            # ),
            # Output shape: (8, 8, 1024)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Conv2D(
                1024,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Latent output (No activation)
            # layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(2 * latent_dim, kernel_initializer=init),
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
