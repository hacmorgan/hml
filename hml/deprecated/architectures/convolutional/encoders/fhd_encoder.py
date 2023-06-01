"""
Encoder model

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import (
    Conv2dBlock,
)


class Encoder(tf.keras.layers.Layer):
    """
    Convolutional encoder

    Encodes images as a multidimensional probability distribution, to form part of a
    variational autoencoder.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        input_shape: Tuple[int, int, int] = (1152, 2048, 3),
        conv_filters: int = 128,
    ) -> "Encoder":
        """
        Construct the encoder

        Args:
            latent_dim: Dimension of posterior distribution generated by encoder
            input_shape: Shape of inputs to encoder
            conv_filters: Number of filters in each non-output conv layer

        Returns:
            Constructed encoder
        """
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=input_shape)
        self.conv1 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 576, 1024
        self.conv2 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 288, 512
        self.conv3 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 144, 256
        self.conv4 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 72, 128
        self.conv5 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 36, 64
        self.conv6 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 18, 32
        self.conv7 = Conv2dBlock(
            filters=conv_filters,
            kernel_size=5,
            strides=(2, 2),
            regularise=0,
            batch_norm=False,
        )
        # Output shape: 9, 16
        self.flattened = layers.Flatten()
        self.loc = layers.Dense(units=latent_dim, activation=None)
        self.scale = layers.Dense(units=latent_dim, activation=tf.nn.softplus)

    def call(self, inputs: tf.Tensor, training=False):
        """
        Produce a posterior distribution when called

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

        # Generate distribution location and scale from dense layers
        loc = self.loc(hf, training=training)
        scale = self.scale(hf, training=training)

        # Construct posterior
        self.posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale
        )

        return self.posterior