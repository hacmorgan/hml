from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import Conv2dBlock, DenseBlock


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 3)):
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=input_shape)
        self.conv1 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        self.conv2 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        self.conv3 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        self.conv4 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        self.conv5 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        self.flattened = layers.Flatten()
        self.encoding = DenseBlock(units=10, activation=tf.nn.sigmoid)
        self.prediction = DenseBlock(
            units=1, activation=tf.nn.sigmoid, useBN=False, drop_prob=0
        )

    def call(self, inputs, training=False):
        he = self.encode(inputs, training=training)
        hp = self.prediction(he, training=training)
        return hp

    def encode(self, inputs, training=False):
        h0 = self.input_(inputs)
        h1 = self.conv1(h0, training=training)
        h2 = self.conv2(h1, training=training)
        h3 = self.conv3(h2, training=training)
        h4 = self.conv4(h3, training=training)
        h5 = self.conv5(h4, training=training)
        hf = self.flattened(h5, training=training)
        he = self.encoding(hf, training=training)
        return he


# def model(input_shape: Tuple[int, int, int] = (64, 64, 3)) -> tf.keras.Sequential:
#     """
#     An encoder based on the DCGAN paper's discriminator
#     """
#     init = tf.keras.initializers.RandomNormal(stddev=0.02)
#     architecture = tf.keras.Sequential(
#         [
#             # Input
#             layers.InputLayer(input_shape=input_shape),
#             layers.Conv2D(
#                 128,
#                 kernel_size=5,
#                 strides=2,
#                 padding="same",
#                 kernel_initializer=init,
#             ),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(alpha=0.2),
#             # Output shape: (64, 64, 128)
#             layers.Conv2D(
#                 128,
#                 kernel_size=5,
#                 strides=2,
#                 padding="same",
#                 kernel_initializer=init,
#             ),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(alpha=0.2),
#             # Output shape: (32, 32, 128)
#             layers.Conv2D(
#                 128,
#                 kernel_size=5,
#                 strides=2,
#                 padding="same",
#                 kernel_initializer=init,
#             ),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(alpha=0.2),
#             # Output shape: (16, 16, 128)
#             layers.Conv2D(
#                 128,
#                 kernel_size=5,
#                 strides=2,
#                 padding="same",
#                 kernel_initializer=init,
#             ),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(alpha=0.2),
#             # Output shape: (8, 8, 128)
#             layers.Conv2D(
#                 128,
#                 kernel_size=5,
#                 strides=2,
#                 padding="same",
#                 kernel_initializer=init,
#             ),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(alpha=0.2),
#             # Output shape: (4, 4, 128)
#             layers.Flatten(),
#             layers.Dense(1),
#             layers.Activation("sigmoid"),
#             # Discriminator output
#         ]
#     )

#     architecture.build()
#     architecture.summary()

#     return architecture
