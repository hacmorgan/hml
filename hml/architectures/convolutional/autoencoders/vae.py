"""
Variational Autoencoder model

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from hml.architectures.convolutional.decoders.fhd_decoder import Decoder
from hml.architectures.convolutional.encoders.fhd_encoder import Encoder


class VariationalAutoEncoder(tf.keras.layers.Layer):
    """
    Variational Autoencoder network
    """

    def __init__(
        self, latent_dim: int = 100, input_shape: Tuple[int, int, int] = (64, 64, 3)
    ) -> "VAE":
        """
        Construct the autoencoder

        Args:
            latent_dim: Dimension of latent distribution (i.e. size of encoder input)
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.input_shape_ = input_shape
        self.encoder_ = Encoder(latent_dim=self.latent_dim_, input_shape=input_shape)
        self.decoder_ = Decoder(latent_dim=self.latent_dim_)
        self.structure = ["encoder_", "decoder_"]

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tfp.distributions.MultivariateNormalDiag, tf.Tensor]:
        """
        Run the model

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Posterior distribution (from encoder outputs)
            Reconstructed image
        """
        posterior = self.encoder_(inputs=inputs, training=training)
        z = posterior.sample()
        reconstruction = self.decoder_(z)
        return posterior, reconstruction
