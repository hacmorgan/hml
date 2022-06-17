#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterator, List, Optional, Tuple

import argparse
import datetime
import os
import random
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# import PIL.ImageTk
import tensorflow as tf

from hml.data_pipelines.utils import (
    load_image,
    insert_image,
    find_training_images,
    normalise,
)


class UpscaleDataset:
    """
    Data serving class for generating large images by "collaging" smaller ones
    """

    def __init__(
        self,
        dataset_path: str,
        output_shape: Tuple[int, int],
        num_examples: int,
        flip_probability: float = 0.3,
    ) -> "UpscaleDataset":
        """
        Construct the data generator.

        Args:
            dataset_path: Path to directory tree containing training data
            output_shape: Desired shape of collage images (rows, cols)
            num_examples: Number of examples to randomly generate per epoch
            flip_probability: Probability of flipping last image vertically/horizontally
        """
        self.dataset_path_ = dataset_path
        self.output_shape_ = output_shape
        self.num_examples_ = num_examples
        self.flip_prob_ = flip_probability
        self.dataset_ = []
        self.have_seen_full_dataset_ = False

    def __call__(self) -> Iterator[np.ndarray]:
        """
        Allow the data generator to be called (by tensorflow) to yield training examples.

        Yields:
            Training examples
        """
        source_images = self.full_dataset()
        for _ in range(self.num_examples_):
            new_image = np.full(
                shape=self.output_shape_, fill_value=-1, dtype=np.float32
            )
            while (free_idx := next_free_pixel(new_image)) is not None:
                try:
                    src_img = next(source_images)
                except StopIteration:
                    source_images = self.full_dataset()
                    continue
                y, x = free_idx
                h, w, _ = src_img.shape
                new_image = insert_image(
                    full_image=new_image, src_image=src_img, location=(y, x)
                )
                # if should_flip := random.random() < self.flip_prob_ / 2:
                #     new_image = insert_image(
                #         full_image=new_image,
                #         src_image=np.fliplr(src_img),
                #         location=(y, x + w),
                #     )
                # elif should_flip < self.flip_prob_:
                #     new_image = insert_image(
                #         full_image=new_image,
                #         src_image=np.flipud(src_img),
                #         location=(y + h, x),
                #     )
            print(new_image.shape)
            yield new_image

    def full_dataset(self) -> Iterator[np.ndarray]:
        """
        Return a generator of all images in the dataset

        Yields:
            Training examples
        """
        # If we have already loaded the dataset, shuffle it and return that
        if self.have_seen_full_dataset_:
            random.shuffle(self.dataset_)
            return iter(self.dataset_)

        # Otherwise load from disk
        for image_path in find_training_images(dataset_path=self.dataset_path_):
            image_np = load_image(image_path=image_path)
            normalised = normalise(image_np)
            self.dataset_.append(normalised)
            yield normalised
        self.have_seen_full_dataset_ = True


def next_free_pixel(image: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Find the coordinates of the next unfilled pixel in an image

    Args:
        image: Image as numpy array

    Returns:
        y coordinate (None if image is full)
        x coordinate (None if image is full)
    """
    ys, xs, *_ = np.where(image == -1)
    if len(ys) == 0:
        return None
    return ys[0], xs[0]
