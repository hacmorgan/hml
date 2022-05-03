from typing import Optional

import tensorflow as tf

from hml.architectures.convolutional.decoders import avae_decoder, avae_decoder_256
from hml.architectures.convolutional.encoders import avae_encoder, avae_encoder_256


class AVAE(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper

    The adversarial part of this model is handled in hml.models.pixel_art_avae, so this
    is just a plain VAE
    """

    def __init__(self, latent_dim: int = 100) -> "AVAE":
        """
        Construct the autoencoder

        Args:
            latent_dim: Dimension of latent distribution (i.e. size of encoder input)
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.encoder_ = avae_encoder.model(latent_dim=self.latent_dim_)
        self.decoder_ = avae_decoder.model(latent_dim=self.latent_dim_)

    def call(self, input_image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Run the model (for visualisation)
        """
        mean, logvar = self.encode(input_image)
        z = self.reparameterize(mean, logvar)
        return self.sample(z)

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
