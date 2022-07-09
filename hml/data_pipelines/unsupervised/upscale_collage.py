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
import shutil
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# import PIL.ImageTk
import tensorflow as tf

from hml.data_pipelines.utils import (
    crop_randomly,
    find_training_images,
    insert_image,
    load_image,
    normalise,
    normalise_tanh,
    save_image,
)
from hml.util.timestamp import iso_condensed


class UpscaleDataset:
    """
    Data serving class for generating large images by "collaging" smaller ones
    """

    def __init__(
        self,
        dataset_path: str,
        output_shape: Tuple[int, int],
        num_examples: int,
        flip_probability: float = 0.5,
        crop_fom_source_images: bool = True,
        save_dir: Optional[str] = None,
    ) -> "UpscaleDataset":
        """
        Construct the data generator.

        Args:
            dataset_path: Path to directory tree containing training data
            output_shape: Desired shape of collage images (rows, cols)
            num_examples: Number of examples to randomly generate per epoch
            flip_probability: Probability of flipping last image vertically/horizontally
            crop_fom_source_images: Randomly crop source images before collaging if True,
                collage full-size images otherwise
            save_dir: If given, save images under this directory to facilitate resuming
                training later with exactly the same images (see also: resume)
        """
        self.dataset_path_ = dataset_path
        self.output_shape_ = output_shape
        self.num_examples_ = num_examples
        self.flip_prob_ = flip_probability
        self.crop_from_source_images_ = crop_fom_source_images
        self.save_dir_ = save_dir
        self.dataset_ = []
        self.have_seen_full_dataset_ = False
        self.first_pass_ = True

        if self.save_dir_ is not None:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(self.save_dir_)

    def __call__(self) -> Iterator[np.ndarray]:
        """
        Allow the data generator to be called (by tensorflow) to yield training examples

        Yields:
            Training examples
        """
        source_images = self.full_dataset()
        end_batch = False
        for _ in range(self.num_examples_):
            new_image = np.full(shape=self.output_shape_, fill_value=-1.0)
            while (free_idx := next_free_pixel(new_image)) is not None:
                try:
                    src_img = next(source_images)
                    if self.crop_from_source_images_:
                        src_img = crop_randomly(src_img)
                except StopIteration:
                    if end_batch:
                        raise RuntimeError("Iterator yielded no items")
                    source_images = self.full_dataset()
                    end_batch = True
                    continue
                end_batch = False
                y, x = free_idx
                h, w, _ = src_img.shape
                new_image = insert_image(
                    full_image=new_image, src_image=src_img, location=(y, x)
                )
            if random.random() < self.flip_prob_:
                new_image = np.fliplr(new_image)
            yield new_image
            if self.first_pass_:
                save_image(
                    image=(new_image * 255).astype(np.uint8),
                    save_path=os.path.join(self.save_dir_, iso_condensed()) + ".png",
                )
        self.first_pass_ = False

    def full_dataset(self) -> Iterator[np.ndarray]:
        """
        Return a generator of all images in the dataset

        Yields:
            Training examples
        """
        # If we have already loaded the dataset, shuffle it and return that
        if self.have_seen_full_dataset_:
            random.shuffle(self.dataset_)
            for image in self.dataset_:
                yield image

        # Otherwise load from disk
        for image_path in find_training_images(dataset_path=self.dataset_path_):
            image_np = load_image(image_path=image_path)
            if image_np is None:
                continue
            normalised = normalise(image_np)
            # normalised = normalise_tanh(image_np)
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


def get_args() -> argparse.Namespace:
    """
    Define and parse command line arguments

    Returns:
        Argument values as argparse namespace
    """
    parser = argparse.ArgumentParser(
        "Visualise training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to dataset directory, containing training images",
    )
    return parser.parse_args()


def cli_main(args: argparse.Namespace) -> int:
    """
    Main CLI routine

    Args:
        args: Command line arguments

    Returns:
        Exit status
    """
    ds = UpscaleDataset(
        dataset_path=args.dataset, output_shape=(1080, 1920, 3), num_examples=2
    )
    print(*ds())
    print(*ds())
    print(len(list(ds.full_dataset())))
    print(len((ds.dataset_)))
    random.shuffle(ds.dataset_)
    print(*iter(ds.dataset_))


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
