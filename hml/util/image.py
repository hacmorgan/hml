#!/usr/bin/env python3


"""
Image utilities

Author:  Hamish Morgan
Date:    19/06/2022
License: BSD
"""


import numpy as np
import PIL.Image
import PIL.ImageEnhance
import tensorflow as tf
import tensorflow_io as tfio


def variance_of_laplacian(images: tf.Tensor, ksize: int = 7, log: bool = True) -> float:
    """
    Compute the variance of the Laplacian (2nd derivative) of an images, as a measure of
    images sharpness.

    This can be used to provide a loss contribution that maximizes images sharpness.

    Args:
        images: Mini-batch of images as 4D tensor
        ksize: Size of Laplace operator kernel
        log: Return log(variance) if True, just variance otherwise

    Returns:
        Variance of the laplacian of images.
    """
    gray_images = tf.image.rgb_to_grayscale(images)
    laplacian = tfio.experimental.filter.laplacian(gray_images, ksize=ksize)
    variance = tf.math.reduce_variance(laplacian)
    if log:
        return tf.math.log(variance)
    return variance


def contrast(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Adjust contrast of image

    Args:
        image: Image to adjust contrast of
        factor: Contrast adjustment direction and intensity. 1 gives unmodified
            input image, < 1 decreases contrast, > 1 increases contrast

    Returns:
        Image with contrast adjusted
    """
    im = PIL.Image.fromarray(image)
    enhancer = PIL.ImageEnhance.Contrast(im)
    return np.array(enhancer.enhance(factor))


def sharpness(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Adjust sharpness of image

    Args:
        image: Image to adjust sharpness of
        factor: Sharpness adjustment direction and intensity. 1 gives unmodified
            input image, < 1 decreases sharpness, > 1 increases sharpness

    Returns:
        Image with sharpness adjusted
    """
    im = PIL.Image.fromarray(image)
    enhancer = PIL.ImageEnhance.Sharpness(im)
    return np.array(enhancer.enhance(factor))
