#!/usr/bin/env python3


"""
Data loader for flood-fill pixel-art generator
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

import tensorflow as tf


def pad_image(image: np.ndarray, pad_width: int) -> np.ndarray:
    """
    Pad the input image with zeros around all sides

    Args:
        image: Unpadded input image
        pad_width: Thickness of outer padding

    Returns:
        Padded input image
    """
    height, width, channels = image.shape
    padded_image_shape = (height + pad_width, width + pad_width, channels)
    padded_image = np.zeros(shape=padded_image_shape, dtype=image.dtype)
    padded_image[
        pad_width : pad_width + height, pad_width : pad_width + width, :
    ] = image
    return padded_image


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image with pixel values on [0, 255] to [-1, 1]
    """
    return image / 255.0


def pad_and_yield_crops(
    image: np.ndarray, shape: Tuple[int, int, int], flip_x: bool = True
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield training data and labels for FloodVAE by convolving across large images

    Images are padded internally, so should not be padded when passed into this
    function.

    For a 3-channel image as in the diagram below, one training input yielded will be
    the context blocks, stacked as (1, 2, 3) to form a 9-channel raster, and the label
    will be block 4.

    Horizontal flipping is also available, where additionally, blocks (5, 4, 3) will be
    stacked and horizontally flipped as a training input, with block 0 horizontally
    flipped as the label.

    Blocks labelled ':' designate the remaining blocks in the image.

    ---------
    |2|3|4|:|
    ---------
    |1|0|5|:|
    ---------
    |:|:|:|:|
    ---------

    Args:
        image: Image to make crops from
        shape: Desired size of crops (rows, cols)
        flip_x: For columns after the leftmost column, yield a horizontally flipped

    Yields:
        Input context blocks, stacked (training data)
        Desired output block (label)
    """
    (image_height, image_width, _) = image.shape
    _, crop_width, _ = shape
    image_padded = pad_image(image, pad_width=crop_width)
    first_block = True
    for x in range(crop_width, image_width - crop_width, crop_width):
        for y in range(crop_width, image_height - crop_width, crop_width):
            # Skip first block, we have no good context
            if first_block:
                first_block = False
                continue
            blocks = {
                0: image_padded[y : y + crop_width, x : x + crop_width, :],
                1: image_padded[y : y + crop_width, x - crop_width : x, :],
                2: image_padded[y - crop_width : y, x - crop_width : x, :],
                3: image_padded[y - crop_width : y, x : x + crop_width, :],
                4: image_padded[
                    y - crop_width : y, x + crop_width : x + 2 * crop_width, :
                ],
                5: image_padded[
                    y : y + crop_width, x + crop_width : x + 2 * crop_width, :
                ],
            }
            outputs = [[np.dstack([blocks[idx] for idx in (1, 2, 3)]), blocks[0]]]
            if flip_x:
                outputs.append(
                    [
                        np.fliplr(array)
                        for array in [
                            np.dstack([blocks[idx] for idx in (5, 4, 3)]),
                            blocks[0],
                        ]
                    ]
                )
            yield from (
                tuple(normalise(raster) for raster in (train_input, label))
                for train_input, label in outputs
            )


class PixelArtFloodDataset:
    """
    Data serving class for pixel art images
    """

    def __init__(
        self,
        dataset_path: str,
        crop_shape: Tuple[int, int, int],
        flip_x: bool = True,
    ) -> "PixelArtFloodDataset":
        """
        Construct the data generator.

        Args:
            dataset_path: Path to directory tree containing training data
            crop_shape: Desired size of crops (rows, cols)
            flip_x: Augment dataset by flipping columns after the leftmost
        """
        self.dataset_path_ = dataset_path
        self.crop_shape_ = crop_shape
        self.flip_x_ = flip_x

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
                print(f"Unusual image found: {image_path}, has shape {image_np.shape}")
                continue
            if image_np.shape[2] > 3:
                image_np = image_np[:, :, :3]
            yield from pad_and_yield_crops(
                image_np, shape=self.crop_shape_, flip_x=self.flip_x_
            )

    def find_training_images(self) -> Iterable[str]:
        """
        Search the dataset.

        A single datum should consist of a directory containing one or more raster
        files. The directory structure above that is arbitrary.

        Yields:
            Path to full size training images
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
