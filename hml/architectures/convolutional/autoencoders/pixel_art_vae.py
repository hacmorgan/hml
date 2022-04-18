import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.decoders import (
    pixel_art_vae_decoder,
)
from hml.architectures.convolutional.encoders import (
    pixel_art_vae_encoder,
)


class PixelArtVAE(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper
    """

    def __init__(self, latent_dim: int = 10) -> "PixelArtVAE":
        """
        Construnt the autoencoder
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.encoder_ = pixel_art_vae_encoder.model(latent_dim=self.latent_dim_)
        self.decoder_ = pixel_art_vae_decoder.model(latent_dim=self.latent_dim_)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, input_image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Run the model (for training)
        """
        z_mean, z_log_var, z = self.encoder_(input_image, training=training)
        decoded = self.decoder_(z, training=training)
        return decoded
