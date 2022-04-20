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
            layers.InputLayer(input_shape=(64, 64, 3)),
            # Input size conv layer (output shape: 32, 32, 128)
            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Half input size conv layer (output shape: 16, 16, 256)
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                256,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Quarter input size conv layer (output shape: 8, 8, 512)
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                512,
                kernel_size=5,
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer=init,
            ),
            # Eighth input size conv layer (output shape: 4, 4, 1024)
            # layers.Dropout(0.3),
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
            # layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(2 * latent_dim, kernel_initializer=init),
            # BatchNorm so that noise can be input at generation time
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
