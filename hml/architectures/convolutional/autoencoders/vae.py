from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from hml.architectures.convolutional.decoders.fhd_decoder import Decoder, decoder
from hml.architectures.convolutional.encoders.fhd_encoder import Encoder, encoder


class Net(tf.keras.layers.Layer):
    """
    Autoencoder network

    The adversarial part of this model is handled in hml.models.pixel_art_avae, so this
    is just a plain VAE
    """

    def __init__(
        self, latent_dim: int = 100, input_shape: Tuple[int, int, int] = (64, 64, 3)
    ) -> "VAE":
        """
        Construct the autoencoder

        Args::waiting
            latent_dim: Dimension of latent distribution (i.e. size of encoder input)
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.input_shape_ = input_shape
        # self.encoder_ = Encoder(latent_dim=self.latent_dim_, input_shape=input_shape)
        self.encoder_ = encoder(latent_dim=self.latent_dim_, input_shape=input_shape)
        # self.decoder_ = Decoder(latent_dim=self.latent_dim_)
        self.decoder_ = decoder(latent_dim=self.latent_dim_)

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tfp.distributions.MultivariateNormDiag, tf.tensor]:
        """
        Run the model

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Posterior distribution (from encoder outputs)
            Reconstructed image
        """
        mean, logvar = self.encode(inputs=inputs, training=training)
        posterior = tfp.distributions.MultivariateNormDiag(loc=mean, scale=logvar)
        z = posterior.sample()
        reconstruction = self.sample(z)
        return posterior, reconstruction

    def custom_compile(
        self, optimizer: tf.keras.optimizers.Optimizer, loss_fn: str = "kl_mse"
    ) -> None:
        """
        Compile the model
        """
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def sample(self, eps: Optional[tf.Tensor] = None, samples: int = 100):
        """
        Run a sample from an input distribution through the decoder to generate an image

        Args:
            eps: Sample from distribution
        """
        if eps is None:
            eps = tf.random.normal(shape=(samples, self.latent_dim_))
        return self.decode(eps, apply_activation=True)

    def encode(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate encodings of images, as mean and log variance vectors that describe a
        multivariate normal diagonal distribution

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Mean of multivariate disctribution
            Log variance of multivariate distribution
        """
        mean, logvar = tf.split(self.encoder_(inputs), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z: tf.Tensor, apply_activation=False):
        """
        Pass a sample form a
        """
        logits = self.decoder_(z)
        if apply_activation:
            probs = tf.sigmoid(logits)
            return probs
        return logits
