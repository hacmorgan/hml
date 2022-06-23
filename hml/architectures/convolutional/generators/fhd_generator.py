"""
Convolutional generator network

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


def generator(
    latent_dim: int = 256,
    conv_filters: int = 128,
) -> tf.keras.models.Model:
    """ """
    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            layers.InputLayer(input_shape=latent_dim),
            *dense_block(
                neurons=(9 * 16 * conv_filters),
                kernel_initializer=kernel_initializer,
                use_bias=False,
            ),
            layers.Reshape((9, 16, conv_filters)),
            # Output shape: (9, 16, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (18, 32, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (36, 64, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (72, 128, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (144,  256, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (288,  512, conv_filters)
            *deconv_2d_block(
                filters=conv_filters, kernel_initializer=kernel_initializer
            ),
            # Output shape: (576, 1024, conv_filters)
            *deconv_2d_block(filters=3, activation="sigmoid"),
            # Output shape: (1152, 2048, 3)
        ]
    )
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
