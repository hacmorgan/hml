from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import (
    Conv2dBlock,
    DenseBlock,
    conv_2d_block,
    dense_block,
)


def encoder(latent_dim: int, input_shape: Tuple[int, int, int]) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Latent input
            layers.InputLayer(input_shape=input_shape),
            *conv_2d_block(filters=128),
            # Output shape: (576, 1024, 128)
            *conv_2d_block(filters=128),
            # Output shape: (288, 512, 128)
            *conv_2d_block(filters=128),
            # Output shape: (144, 256, 128)
            *conv_2d_block(filters=128),
            # Output shape: (72, 128, 128)
            *conv_2d_block(filters=128),
            # Output shape: (36, 64, 128)
            *conv_2d_block(filters=128),
            # Output shape: (18, 32, 128)
            *conv_2d_block(filters=128),
            # Output shape: (9, 16, 128)
            layers.Flatten(),
            *dense_block(neurons=latent_dim * 2, activation="", batch_norm=False),
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, latent_dim: int = 256, input_shape: Tuple[int, int, int] = (1152, 2048, 3)
    ):
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=input_shape)
        self.conv1 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 576, 1024
        self.conv2 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 288, 512
        self.conv3 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 144, 256
        self.conv4 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 72, 128
        self.conv5 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 36, 64
        self.conv6 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 18, 32
        self.conv7 = Conv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 9, 16
        self.flattened = layers.Flatten()
        self.encoding = DenseBlock(
            units=2 * latent_dim, activation=None, drop_prob=0, useBN=False
        )

    def call(self, inputs, training=False):
        h0 = self.input_(inputs)
        h1 = self.conv1(h0, training=training)
        h2 = self.conv2(h1, training=training)
        h3 = self.conv3(h2, training=training)
        h4 = self.conv4(h3, training=training)
        h5 = self.conv5(h4, training=training)
        h6 = self.conv6(h5, training=training)
        h7 = self.conv7(h6, training=training)
        hf = self.flattened(h5, training=training)
        he = self.encoding(hf, training=training)
        return he
