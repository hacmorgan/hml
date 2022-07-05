"""
Convolutional generator network

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


from typing import Tuple

import sys

import tensorflow as tf

from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import (
    Deconv2dBlock,
    DenseBlock,
    deconv_2d_block,
    dense_block,
)


LATENT_SHAPE_SQUARE = (8, 8)
LATENT_SHAPE_WIDE = (9, 16)
LATENT_SHAPE_ULTRAWIDE = (9, 32)


def generator(
    output_shape: Tuple[int, int, int],
    latent_dim: int = 128,
    conv_filters: int = 128,
    latent_shape: Tuple[int, int] = LATENT_SHAPE_WIDE,
    strides: int = 2,
) -> tf.keras.models.Model:
    """
    Convolutional generator

    Adds fractionally strided convolutions until the desired output size is met
    """
    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)

    # Start with input and dense layers
    network_layers = [
        layers.InputLayer(input_shape=latent_dim),
        *dense_block(
            neurons=(latent_shape[0] * latent_shape[1] * conv_filters),
            kernel_initializer=kernel_initializer,
            bias=False,
            batch_norm=False,
            drop_prob=0,
        ),
        layers.Reshape(latent_shape + (conv_filters,)),
    ]
    shape = list(latent_shape)

    # Add fractionally strided convs until output feature map is half the size of output
    while shape[0] < output_shape[0] / strides and shape[1] < output_shape[1] / strides:
        network_layers += deconv_2d_block(
            filters=conv_filters,
            kernel_initializer=kernel_initializer,
            bias=False,
            batch_norm=False,
            drop_prob=0,
            strides=strides,
        )
        shape[0] *= strides
        shape[1] *= strides

    # Add output layer
    network_layers += deconv_2d_block(
        filters=3,
        activation=tf.nn.sigmoid,
        kernel_initializer=kernel_initializer,
        bias=False,
        batch_norm=False,
        drop_prob=0,
        strides=strides,
    )
    shape[0] *= strides
    shape[1] *= strides
    if tuple(shape) != output_shape[:2]:
        print(
            f"Output shape not achievable, requested {output_shape}, achieved {shape}",
            file=sys.stderr,
        )

    # Build the network
    architecture = tf.keras.Sequential(network_layers)
    architecture.build()
    architecture.summary()
    return architecture


class Generator(tf.keras.models.Model):
    """
    Convolutional generator
    """

    def __init__(self, latent_dim: int = 256, conv_filters: int = 128) -> "Generator":
        """
        Construct the generator

        Args:
            latent_dim: Dimension of latent input to the decoder
            conv_filters: Number of filters in each non-output conv layer

        Returns:
            Constructed generator
        """
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=latent_dim)
        self.dense = DenseBlock(
            units=(9 * 16 * conv_filters),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        self.reshape = layers.Reshape((9, 16, conv_filters))
        self.conv1 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        # Output shape: 18, 32
        self.conv2 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        # Output shape: 36, 64
        self.conv3 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        # Output shape: 72, 128
        self.conv4 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        # Output shape: 144, 256
        self.conv5 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
        )
        # Output shape: 288, 512
        self.conv6 = Deconv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            # batch_norm=False,
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
            batch_norm=False,
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

    def custom_compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        """
        Compile the model
        """
        super().compile()
        self.optimizer = optimizer
