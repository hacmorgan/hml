"""
Encoder model

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


from typing import Tuple

import tensorflow as tf

from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import (
    Deconv2dBlock,
    DenseBlock,
    deconv_2d_block,
    dense_block,
)


class Decoder(tf.keras.layers.Layer):
    """
    Convolutional decoder
    """

    def __init__(self, latent_dim: int = 256) -> "Decoder":
        """
        Construct the decoder

        Args:
            latent_dim: Dimension of posterior distribution generated by encoder

        Returns:
            Constructed decoder
        """
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=latent_dim)
        self.dense = DenseBlock(
            units=(9 * 16 * 128),
            regularise=0,
        )
        self.reshape = layers.Reshape((9, 16, 128))
        self.conv1 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 18, 32
        self.conv2 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 36, 64
        self.conv3 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 72, 128
        self.conv4 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 144, 256
        self.conv5 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 288, 512
        self.conv6 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
        )
        # Output shape: 576, 1024
        self.conv7 = Deconv2dBlock(
            filters=3,
            kernel_size=5,
            strides=(2, 2),
            activation=tf.nn.sigmoid,
            # activation=None,
            drop_prob=0,
            regularise=0,
            useBN=False,
        )
        # Output shape: 1152, 2048

    def call(self, inputs, training=False):
        """
        Generate an image when called

        Args:
            inputs: Sample(s) from posterior distribution
            training: True if we are training, False otherwise

        Returns:
            Generated image
        """
        h0 = self.input_(inputs)
        hd = self.dense(h0, training=training)
        hr = self.reshape(hd, training=training)
        h1 = self.conv1(hr, training=training)
        h2 = self.conv2(h1, training=training)
        h3 = self.conv3(h2, training=training)
        h4 = self.conv4(h3, training=training)
        h5 = self.conv5(h4, training=training)
        h6 = self.conv6(h5, training=training)
        h7 = self.conv7(h6, training=training)
        return h7
