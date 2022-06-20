#!/usr/bin/env python3


"""
Generic data loading utilities
"""


__author__ = "Hamish Morgan"


from typing import Callable, Iterator, Optional, Tuple

import os
import random

import numpy as np
import PIL.Image


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image with pixel values on [0, 255] to [0, 1]
    """
    return image / 255


def normalise_tanh(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image with pixel values on [0, 255] to [-1, 1]
    """
    return (image - 127.5) / 127.5


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk, catching common issues
    """
    try:
        with PIL.Image.open(image_path) as image:
            image_np = np.array(image)
    except PIL.UnidentifiedImageError:
        print(f"Cannot open file: {image_path}, it will not be used for training")
    if len(image_np.shape) == 2:
        image_np = np.array(image.convert("RGB"))
    elif len(image_np.shape) != 3:
        print(f"Unusual image found: {image_path}, has shape {image_np.shape}")
        return None
    if image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]
    return image_np


def walk_using_scandir(path: str, shuffle: bool = True) -> Iterator[os.DirEntry]:
    """
    Find all files under a top-level path using os.scandir recursively

    This is much faster than even `os.walk(..., topdown=False)` when walking slow
    (remote) filesystems, like s3fs for AWS s3.

    Args:
        path: Top level path to walk
        shuffle: Shuffle directory elements before recursing/yielding if True, keep in
            order of discovery otherwise

    Yields:
        Path to every file under path in the filesystem
    """
    dir_entries = os.scandir(path)
    if shuffle:
        dir_entries = list(dir_entries)
        random.shuffle(dir_entries)
    for entry in dir_entries:
        if entry.is_dir():
            yield from walk_using_scandir(entry.path)
        else:
            yield entry.path


def find_training_images(
    dataset_path: str, filter_fn: Optional[Callable] = None
) -> Iterator[str]:
    """
    Search the dataset.

    A single datum should consist of a directory containing one or more raster files.
    The directory structure above that is arbitrary.

    Args:
        dataset_path: Path to root of dataset
        filter_fn: Optional function (str -> bool) to filter out/in filepaths

    Returns:
        Iterator of images
    """
    if filter_fn is None:
        filter_fn = lambda filename: True
    return filter(filter_fn, walk_using_scandir(dataset_path))


def insert_image(
    full_image: np.ndarray, src_image: np.ndarray, location: Tuple[int, int]
) -> np.ndarray:
    """
    Paste src_image into full_image at location

    Handles cropping when src_image is bigger than the amount of room to paste it.

    Args:
        full_image: Full size image, prefilled with -1s
        src_image: Image to be pasted into full-size canvas
        location: Where to paste src_image (y, x)

    Returns:
        Full size image with source image pasted
    """
    y, x = location
    h, w, _ = src_image.shape
    H, W, _ = full_image.shape
    vspace = H - y
    hspace = W - x
    if vspace < h:
        h = vspace
        src_image = src_image[:h, :, :]
    if hspace < w:
        w = hspace
        src_image = src_image[:, :w, :]
    full_image[y : y + h, x : x + w, :] = src_image
    return full_image
