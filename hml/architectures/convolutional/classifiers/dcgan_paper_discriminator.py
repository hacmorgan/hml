import tensorflow as tf
from tensorflow.keras import layers


def model() -> tf.keras.Sequential:
    """
    A guess at what the DCGAN paper's discriminator looked like
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Input size conv layer (output shape: 64, 64, 64)
            layers.Conv2D(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                input_shape=(64, 64, 3),
                kernel_initializer=init,
            ),
            layers.LeakyReLU(alpha=0.2),
            # Half input size conv layer (output shape: 32, 32, 128)
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                128, kernel_size=5, strides=2, padding="same", kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Quarter input size conv layer (output shape: 16, 16, 256)
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                256, kernel_size=5, strides=2, padding="same", kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Eighth input size conv layer (output shape: 8, 8, 512)
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                512, kernel_size=5, strides=2, padding="same", kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Sixteenth input size conv layer (output shape: 4, 4, 1024)
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Conv2D(
                1024, kernel_size=5, strides=2, padding="same", kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            # Output neuron
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(1),
            layers.LeakyReLU(alpha=0.2),
        ]
    )

    architecture.build()
    architecture.summary()

    return architecture
