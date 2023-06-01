#!/usr/bin/env python3


"""
Find images in given directory and resize to desired size
"""


__author__ = "Hamish Morgan"


from typing import Iterable, List, Optional, Tuple

import argparse
import datetime
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# import PIL.ImageTk
import tensorflow as tf

# import tkinter


def crops_from_full_size(
    image: np.ndarray, shape: Tuple[int, int, int]
) -> Iterable[np.ndarray]:
    """
    From a full sized image, yield image crops of the desired size.

    Args:
        image: Image to make crops from
        shape: Desired size of crops (rows, cols)

    Yields:
        Image crops of desired size
    """
    (image_height, image_width, _) = image.shape
    crop_height, crop_width, _ = shape
    for y in (
        *range(0, image_height - crop_height, crop_height),
        image_height - crop_height,
    ):
        for x in (
            *range(0, image_width - crop_width, crop_width),
            image_width - crop_width,
        ):
            yield image[y : y + crop_height, x : x + crop_width, :]


def permute_flips(
    image: np.ndarray, flip_x: bool = True, flip_y: bool = True
) -> List[np.ndarray]:
    """
    From an input image, yield that image flipped in x and/or y
    """
    images = [image]
    if flip_x:
        images.append(np.fliplr(image))
    if flip_y:
        images += [*map(np.flipud, images)]
    return images


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image with pixel values on [0, 255] to [-1, 1]
    """
    return image / 255.0


class ResizeDataset:
    """
    Data serving class for generic images that should be resized (as opposed to cropping from a bigger image).
    """

    def __init__(
        self, dataset_path: str, output_shape: Tuple[int, int, int]
    ) -> "PixelArtDataset":
        """
        Construct the data generator.

        Args:
            dataset_path: Path to directory tree containing training data
            crop_shape: Desired size of crops (rows, cols, channels)
        """
        self.dataset_path_ = dataset_path
        self.output_shape_ = output_shape

    def __call__(self) -> Iterable[np.ndarray]:
        """
        Allow the data generator to be called (by tensorflow) to yield training examples.

        Yields:
            Training examples
        """
        for image_path in self.find_training_images():
            try:
                with PIL.Image.open(image_path) as image:
                    image_np = np.array(image)
                    # image_np = np.array(image.convert("HSV"))
            except PIL.UnidentifiedImageError:
                print(
                    f"Cannot open file: {image_path}, it will not be used for training"
                )
            if len(image_np.shape) != 3:
                image_np = np.array(image.convert("RGB"))
            if image_np.shape[2] > 3:
                image_np = image_np[:, :, :3]
            image_resized = cv2.resize(
                image_np,
                dsize=self.output_shape_[:-1][::-1],
                interpolation=cv2.INTER_AREA,
            )
            # image_resized = image_np
            # yield from map(normalise, permute_flips(image_crop))
            # yield from map(normalise, permute_flips(image_resized, flip_y=False))
            yield normalise(image_resized)

    def find_training_images(self) -> Iterable[str]:
        """
        Search the dataset.

        A single datum should consist of a directory containing one or more raster files. The
        directory structure above that is arbitrary.

        Returns:
            List
        """
        for root, _, filenames in os.walk(self.dataset_path_, followlinks=True):
            for filename in filenames:
                yield os.path.join(root, filename)


def visualise_dataset(
    dataset_path: str,
    train_crop_shape: Tuple[int, int, int] = (28, 28, 3),
    as_tensorflow_dataset: bool = True,
) -> None:
    """
    View the dataset

    Args
        dataset_path: Directory containing training data
        train_crop_shape: Expected shape of crops for training
        as_tensorflow_dataset: Create tensorflow dataset and read from that if True,
                               show raw generator output otherwise.
    """
    if as_tensorflow_dataset:
        train_images = tf.data.Dataset.from_generator(
            PixelArtSigmoidDataset(
                dataset_path=dataset_path, crop_shape=train_crop_shape
            ),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        for image in iter(train_images):
            plt.imshow(image)
    else:
        for image in PixelArtSigmoidDataset(
            dataset_path=dataset_path, crop_shape=train_crop_shape
        )():
            plt.imshow(image)
            plt.show()


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
    main(
        mode=args.mode,
        model_dir=os.path.join("models", args.model_name),
        dataset_path=args.dataset,
        continue_from_checkpoint=args.checkpoint,
        generator_input=args.generator_input,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
