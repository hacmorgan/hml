import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.decoders import avae_decoder
from hml.architectures.convolutional.encoders import avae_encoder
from hml.architectures.convolutional.discriminators import avae_discriminator


class AVAE(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper
    """

    def __init__(self, latent_dim: int = 10) -> "PixelArtVAE":
        """
        Construct the autoencoder
        """
        super().__init__()
        self.latent_dim_ = latent_dim
        self.encoder_ = avae_encoder.model(latent_dim=self.latent_dim_)
        self.decoder_ = avae_decoder.model(latent_dim=self.latent_dim_)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="discriminator_loss"
        )

    def call(self, input_image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Run the model (for visualisation)
        """
        mean, logvar = self.encode(input_image)
        z = self.reparameterize(mean, logvar)
        return self.sample(z)

    @tf.function
    def sample(self, eps=None):
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
