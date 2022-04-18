from typing import Tuple

import tensorflow as tf


class VAESampling(tf.keras.layers.Layer):
    """
    Layer that performs reparameterisation "trick"

    Samples a latent encoding (of some pixel art) from the probability density function
    defined by the mean and log variance output by the encoder.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Genrate a latent encoding from PDF mean and log variance

        Args:
            inputs: Mean and log variance of PDF

        Returns:
            Latent encoding
        """
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        if batch is None:
            batch = 1
        dim = z_mean.shape[1]

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
