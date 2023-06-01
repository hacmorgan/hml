#!/usr/bin/env python3


"""
Loss functions
"""


__author__ = "Hamish Morgan"


from typing import Tuple

import tensorflow as tf


# This method returns a helper function to compute binary crossentropy loss
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def compute_generator_loss(
    generated_output: tf.Tensor,
) -> float:
    """
    Compute loss for training the generator

    Args:
        generated_output: Output of discriminator on generated images

    Returns:
        Loss for training generator
    """
    return bce(tf.ones_like(generated_output), generated_output)


def compute_discriminator_loss(
    real_output: tf.Tensor, generated_output: tf.Tensor
) -> Tuple[float, float, float]:
    """
    Compute loss for training discriminator network

    Args:
        real_output: Output of discriminator on real training data
        generated_output: Output of discriminator on generated images

    Returns:
        Total loss
        Loss on real images
        Loss on generated images
    """
    # Compute loss components
    real_loss = bce(tf.ones_like(real_output), real_output)
    generated_loss = bce(tf.zeros_like(generated_output), generated_output)

    # Compute total loss and return
    total_loss = real_loss + generated_loss
    return total_loss, real_loss, generated_loss
