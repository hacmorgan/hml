from typing import Optional, Tuple

import tensorflow as tf

from hml.architectures.convolutional.decoders.fhd_decoder import Decoder, decoder
from hml.architectures.convolutional.encoders.fhd_encoder import Encoder, encoder


class VAE(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper

    The adversarial part of this model is handled in hml.models.pixel_art_avae, so this
    is just a plain VAE
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
        # self.encoder_ = Encoder(latent_dim=self.latent_dim_, input_shape=input_shape)
        self.encoder_ = encoder(latent_dim=self.latent_dim_, input_shape=input_shape)
        # self.decoder_ = Decoder(latent_dim=self.latent_dim_)
        self.decoder_ = decoder(latent_dim=self.latent_dim_)

    def call(self, input_image: tf.Tensor) -> tf.Tensor:
        """
        Run the model (for visualisation)
        """
        mean, logvar = self.encode(input_image)
        z = self.reparameterize(mean, logvar)
        return self.sample(z)

    def compile(
        self, optimizer: "Optimizer", loss_fn: str = "binary_crossentropy"
    ) -> None:
        """
        Compile the model
        """
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def sample(self, eps: Optional[tf.Tensor] = None):
        """
        Run a sample from an input distribution through the decoder to generate an image

        Args:
            eps: Sample from distribution
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim_))
        return self.decode(eps, apply_activation=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder_(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_activation=False):
        logits = self.decoder_(z)
        if apply_activation:
            probs = tf.sigmoid(logits)
            return probs
        return logits
