import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.decoders import pixel_art_decoder
from hml.architectures.convolutional.encoders import pixel_art_encoder


class PixelArtAE(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper
    """

    def __init__(self, latent_dim: int = 100) -> "PixelArtAE":
        """
        Construnt the autoencoder
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.encoder_ = pixel_art_encoder.model(latent_dim=self.latent_dim_)
        self.decoder_ = pixel_art_decoder.model(latent_dim=self.latent_dim_)

    def call(self, input_image: tf.Tensor) -> tf.Tensor:
        """
        Run the model (for training)
        """
        encoded = self.encoder_(input_image)
        decoded = self.decoder_(encoded)
        return decoded
