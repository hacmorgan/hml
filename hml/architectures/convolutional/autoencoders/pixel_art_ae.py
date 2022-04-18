import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.decoders import (
    pixel_art_decoder,
    pixel_art_decoder_sigmoid,
)
from hml.architectures.convolutional.encoders import (
    pixel_art_encoder,
    pixel_art_encoder_tanh,
)


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
        self.encoder_ = pixel_art_encoder_tanh.model(latent_dim=self.latent_dim_)
        self.decoder_ = pixel_art_decoder_sigmoid.model(latent_dim=self.latent_dim_)

    def call(self, input_image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Run the model (for training)
        """
        encoded = self.encoder_(input_image, training=training)
        decoded = self.decoder_(encoded, training=training)
        return decoded
