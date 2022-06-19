#!/usr/bin/env python3


"""
Image utilities

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


import tensorflow as tf
import tensorflow_io as tfio


def variance_of_laplacian(images: tf.Tensor, ksize: int = 7) -> float:
    """
    Compute the variance of the Laplacian (2nd derivative) of an images, as a measure of
    images sharpness.

    This can be used to provide a loss contribution that maximizes images sharpness.

    Args:
        images: Mini-batch of images as 4D tensor
        ksize: Size of Laplace operator kernel

    Returns:
        Variance of the laplacian of images.
    """
    gray_images = tf.image.rgb_to_grayscale(images)
    laplacian = tfio.experimental.filter.laplacian(gray_images, ksize=ksize)
    return tf.math.reduce_variance(laplacian)
