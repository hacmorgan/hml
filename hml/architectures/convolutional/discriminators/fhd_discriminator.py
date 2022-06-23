"""
Convolutional discriminator network

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


from typing import Tuple

from functools import partial

import tensorflow as tf

from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import Conv2dBlock, DenseBlock
from hml.architectures.convolutional.blocks import conv_2d_block, dense_block


def discriminator(
    input_shape: Tuple[int, int, int] = (1152, 2048, 3),
    conv_filters: int = 128,
) -> tf.keras.models.Model:
    """
    Functional API discriminator
    """
    leaky_relu = partial(tf.nn.leaky_relu, alpha=0.2)
    kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (576, 1024, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (288,  512, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (144,  256, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (72, 128, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (36, 64, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (18, 32, conv_filters)
            *conv_2d_block(
                filters=conv_filters,
                activation=leaky_relu,
                kernel_initializer=kernel_initializer,
            ),
            # Output shape: (9, 16, conv_filters)
            layers.Flatten(),
            layers.Dense(
                1, activation=tf.nn.sigmoid, kernel_initializer=kernel_initializer
            ),
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture


class Discriminator(tf.keras.models.Model):
    """
    Convolutional discriminator

    Can produce an encoding of the input in addition to the usual prediction of whether
    it is real or fake, to help facilitate e.g. methods to prevent mode-collapse.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1152, 2048, 3),
        conv_filters: int = 128,
    ) -> "Discriminator":
        """
        Construct the discriminator

        Args:
            input_shape: Shape of inputs to discriminator
            conv_filters: Number of filters in each non-output conv layer

        Returns:
            Constructed discriminator
        """
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=input_shape)
        self.conv1 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 576, 1024
        self.conv2 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 288, 512
        self.conv3 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 144, 256
        self.conv4 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 72, 128
        self.conv5 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 36, 64
        self.conv6 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 18, 32
        self.conv7 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            drop_prob=0,
            activation=tf.nn.leaky_relu,
            # useBN=False,
        )
        # Output shape: 9, 16
        self.flattened = layers.Flatten()
        self.encoding = DenseBlock(
            units=256,
            activation=tf.nn.sigmoid,
            regularise=0,
            drop_prob=0,
            # useBN=False
        )
        self.prediction = layers.Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, inputs: tf.Tensor, training=False):
        """
        Produce a posterior distribution when called

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Posterior distribution (from encoder outputs)
        """
        enc = self.encode(inputs, training=training)
        pred = self.prediction(enc, training=training)
        return pred

    def encode(self, inputs: tf.Tensor, training=False):
        """
        Produce an encoding of the inputs

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Posterior distribution (from encoder outputs)
        """
        # Pass input through conv layers
        h0 = self.input_(inputs)
        h1 = self.conv1(h0, training=training)
        h2 = self.conv2(h1, training=training)
        h3 = self.conv3(h2, training=training)
        h4 = self.conv4(h3, training=training)
        h5 = self.conv5(h4, training=training)
        h6 = self.conv6(h5, training=training)
        h7 = self.conv7(h6, training=training)
        hf = self.flattened(h7, training=training)
        enc = self.encoding(hf, training=training)
        return enc

    def custom_compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        """
        Compile the model
        """
        super().compile()
        self.optimizer = optimizer
