#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterable, Dict, List, Optional, Tuple

import argparse
import datetime
import math
import os
import subprocess
import sys
import time

import matplotlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio
import tensorflow_datasets as tfds


from hml.architectures.convolutional.autoencoders.vae import VAE
from hml.architectures.convolutional.discriminators.avae_discriminator import (
    model as discriminator_model,
)

from hml.data_pipelines.unsupervised.pixel_art_flood import PixelArtFloodDataset

# from hml.data_pipelines.unsupervised.pixel_art_flood_3_block_input import (
#     PixelArtFloodDataset,
# )


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")

UPDATE_TEMPLATE = """
Epoch: {epoch}    Step: {step}    Time: {epoch_time:.2f}
KL loss:                       {kl_loss:>6.4f}
Reconstruction sharpness loss: {reconstruction_sharpness_loss:>6.4f}
Generation sharpness loss:     {generation_sharpness_loss:>6.4f}
Total loss:                    {vae_loss:>6.4f}
Discriminator loss:            {discriminator_loss:>6.4f}
"""
# Discriminator loss: {discriminator_loss}


# This method returns a helper function to compute mean squared error loss
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()


global generator_
global generated_image
global latent_input
global input_neuron
global latent_input_canvas
global generator_output_canvas


def modified_files_in_git_repo() -> bool:
    """
    Ensure hml has no uncommitted files.

    It is required that the git hash is written to the model dir, to ensure the same
    code that was used for training can be retrieved later and paired with the trained
    parameters.

    Returns:
        True if there are modified files, False otherwise
    """
    result = subprocess.run(
        'cd "$HOME/src/hml" && git status --porcelain=v1 | grep -v -e "^??" -e "^M"',
        shell=True,
        stdout=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8").strip()
    if len(output) > 0:
        print(
            """
            ERROR: Uncommitted code in $HOME/src/hml

            Commit changes before re-running train
            """,
            file=sys.stderr,
        )
        return True
    return False


def write_commit_hash_to_model_dir(model_dir: str) -> None:
    """
    Write the commit hash used for training to the model dir
    """
    result = subprocess.run(
        'cd "$HOME/src/hml" && git rev-parse --verify HEAD',
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    commit_hash = result.stdout.decode("utf-8").strip()
    with open(os.path.join(model_dir, "commit-hash"), "w") as hashfile:
        hashfile.write(commit_hash)


def generate_save_image(predictions: tf.Tensor, output_path: str) -> None:
    """
    Actually generate the image and save it.

    This is done in a separate thread to avoid waiting for io.
    """
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        generated_rgb_image = np.array((predictions[i, :, :, :] * 255.0)).astype(
            np.uint8
        )
        # generated_rgb_image = cv2.cvtColor(generated_hsv_image, cv2.COLOR_HSV2RGB)
        plt.imshow(generated_rgb_image)
        plt.axis("off")
    plt.savefig(output_path, dpi=250)


def reproduce_save_image(
    test_images: tf.Tensor,
    test_labels: tf.Tensor,
    reproductions: tf.Tensor,
    output_path: str,
) -> None:
    """
    Regenerate some images and save to file.
    """
    block_width = int(test_images.shape[1])
    outputs = np.zeros((8, test_images.shape[1] * 2 * 3, test_images.shape[2] * 3, 3))
    for i, (test_image, test_label, reproduction) in enumerate(
        zip(test_images, test_labels, reproductions)
    ):
        plt.subplot(2, 4, i + 1)

        # Unstack training inputs into context blocks
        context = np.zeros((3 * block_width, 3 * block_width, 3))
        context[block_width : 2 * block_width, 0:block_width, :] = test_image[..., 0:3]
        context[:block_width, block_width : 2 * block_width, :] = test_image[..., 3:6]
        context[
            block_width : 2 * block_width, 2 * block_width : 3 * block_width, :
        ] = test_image[..., 6:9]
        context[
            2 * block_width : 3 * block_width, block_width : 2 * block_width, :
        ] = test_image[..., 9:12]

        # Fill in the bottom right block with training label or reproduction
        ground_truth = context.copy()
        ground_truth[
            block_width : 2 * block_width, block_width : 2 * block_width, :
        ] = test_label
        reproduced = context.copy()
        reproduced[
            block_width : 2 * block_width, block_width : 2 * block_width, :
        ] = reproduction

        # Stack both and save
        stacked = np.vstack([ground_truth, reproduced])
        outputs[i, ...] = stacked

        rgb_image = np.array((stacked * 255.0)).astype(np.uint8)
        plt.imshow(rgb_image)
        plt.axis("off")
    plt.savefig(output_path, dpi=250)
    return outputs


def generate_and_save_images(
    autoencoder: tf.keras.Sequential,
    epoch: int,
    test_input: tf.Tensor,
    progress_dir: str,
) -> tf.Tensor:
    """
    Generate and save images
    """
    predictions = autoencoder.sample(test_input)
    output_path = os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png")
    # threading.Thread(
    #     target=generate_save_image, args=(predictions, output_path)
    # ).start()
    generate_save_image(predictions, output_path)
    return predictions


def show_reproduction_quality(
    autoencoder: tf.keras.models.Model,
    epoch: int,
    test_images: tf.Tensor,
    test_labels: tf.Tensor,
    reproductions_dir: str,
) -> tf.Tensor:
    """
    Generate and save images
    """
    reproductions = autoencoder.call(test_images)
    print(f"{np.mean(reproductions)=}, {np.std(reproductions)=}")
    print(f"{np.mean(test_images)=}, {np.std(test_images)=}")
    output_path = os.path.join(
        reproductions_dir, f"reproductions_at_epoch_{epoch:04d}.png"
    )
    # threading.Thread(
    #     target=reproduce_save_image, args=(test_images, reproductions, output_path)
    # ).start()
    outputs = reproduce_save_image(test_images, test_labels, reproductions, output_path)
    # return tf.concat([test_labels[:8, ...], reproductions], axis=1)
    return outputs


def save_flood_generated_image(image: np.ndarray, output_dir: str, epoch: int) -> None:
    """
    Save an image
    """
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.axis("off")
    output_path = os.path.join(output_dir, f"flood_generation_at_epoch_{epoch:04d}.png")
    plt.savefig(output_path, dpi=250)


def flood_generate(
    autoencoder: tf.keras.models.Model,
    seed: Optional[tf.Tensor] = None,
    shape: Tuple[int, int] = (4, 4),
) -> tf.Tensor:
    """
    Flood-fill a large image by autogenerating the every other block (like a
    checkerboard), then interpolating the remaining blocks from context

    Args:
        autoencoder: Autoencoder model
        seed: Random normal noise as input to decoder, of shape (num_blocks_to_generate, latent_dim)
        shape: Desired shape (in blocks) of output image

    Returns:
        Generated image
    """
    # Unpack relevant shapes
    y_blocks, x_blocks = shape
    block_width = autoencoder.input_shape_[1]

    # Make blank (0s) canvas, padded
    padded_output_image = np.zeros(
        ((shape[0] + 2) * block_width, (shape[1] + 2) * block_width, 3)
    )

    # Calculate how many blocks we need to generate from noise
    num_generated_blocks = math.ceil(x_blocks * y_blocks / 2)

    # Generate input noise or verify we have enough if passed as arg
    if seed is None:
        seed = tf.random.normal([num_generated_blocks, autoencoder.latent_dim_])
    elif seed.shape[0] < num_generated_blocks:
        raise ValueError(
            "Expected {} latent inputs, only got {}".format(
                num_generated_blocks, seed.shape[0]
            )
        )

    # Generate every other block from noise
    latent_idx = 0
    for y in range(1, y_blocks + 1):
        for x in range(y % 2 + 1, x_blocks + 1, 2):
            padded_output_image[
                y * block_width : (y + 1) * block_width,
                x * block_width : (x + 1) * block_width,
                :,
            ] = autoencoder.sample(tf.expand_dims(seed[latent_idx, :], axis=0))
            latent_idx += 1

    # Fill in the blanks
    for y in range(1, y_blocks + 1):
        for x in range((y + 1) % 2 + 1, x_blocks + 1, 2):
            context_blocks = {
                1: padded_output_image[
                    (y + 0) * block_width : (y + 1) * block_width,
                    (x - 1) * block_width : (x + 0) * block_width,
                    :,
                ],
                2: padded_output_image[
                    (y - 1) * block_width : (y + 0) * block_width,
                    (x + 0) * block_width : (x + 1) * block_width,
                    :,
                ],
                3: padded_output_image[
                    (y + 0) * block_width : (y + 1) * block_width,
                    (x + 1) * block_width : (x + 2) * block_width,
                    :,
                ],
                4: padded_output_image[
                    (y + 1) * block_width : (y + 2) * block_width,
                    (x + 0) * block_width : (x + 1) * block_width,
                    :,
                ],
            }
            context = np.dstack([context_blocks[idx] for idx in (1, 2, 3, 4)])
            interpolated_block = autoencoder.call(tf.expand_dims(context, axis=0))
            padded_output_image[
                (y + 0) * block_width : (y + 1) * block_width,
                (x + 0) * block_width : (x + 1) * block_width,
                :,
            ] = interpolated_block

    # Return image without padding
    return padded_output_image[
        block_width : (y_blocks + 1) * block_width,
        block_width : (x_blocks + 1) * block_width,
        :,
    ]


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


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


def compute_vae_loss(
    vae: tf.keras.models.Model,
    discriminator: tf.keras.Sequential,
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 1e0,
    beta: float = 0e0,
    gamma: float = 2e-1,
    delta: float = 0e0,
    epsilon: float = 1e0,
) -> Tuple[float, tf.Tensor, tf.Tensor, float, float, float]:
    """
    Compute loss for training VAE

    Args:
        vae: VAE model - should have encode(), decode(), reparameterize(), and sample()
             methods
        discriminator: The discriminator model
        images: Minibatch of training images
        labels: Minibatch of training labels
        alpha: Contribution of KL divergence loss to total loss
        beta: Contribution of discriminator loss on reconstructed images to total loss
        gamma: Contribution of discriminator loss on generated images to total loss
        delta: Contribution of generated image sharpness loss to total loss
        epsilon: Contribution of reconstructed image sharpness loss to total loss

    Returns:
        Total loss
        VAE (KL divergence) component of loss
        Discrimination loss on reconstructed images
        Discrimination loss on generated images
        Image sharpness loss on generated images
        Reconstructed images (used again to compute loss for training discriminator)
        Generated images (used again to compute loss for training discriminator)
    """
    # Reconstruct a real image
    mean, logvar = vae.encode(images)
    z = vae.reparameterize(mean, logvar)
    x_logit = vae.decode(z)
    reconstructed = vae.sample(z)

    # Flood generate a 3x3 image from noise input
    z_gen = tf.random.normal(shape=z.shape)
    generated = vae.sample(z_gen)

    # Compute KL divergence loss
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=labels)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

    # Loss from discriminator output on reconstructed image
    reconstruction_output = discriminator(reconstructed)
    discrimination_reconstruction_loss = bce(
        tf.ones_like(reconstruction_output), reconstruction_output
    )

    # Loss from discriminator output on generated image
    generated_output = discriminator(generated)
    discrimination_generation_loss = bce(
        tf.ones_like(generated_output), generated_output
    )

    # Sharpness loss on generated images
    sharpness_loss_generated = -(
        variance_of_laplacian(generated, ksize=3)
        + variance_of_laplacian(generated, ksize=5)
        + variance_of_laplacian(generated, ksize=7)
    )

    # Sharpness loss on reconstructed images
    sharpness_loss_reconstructed = -(
        variance_of_laplacian(reconstructed, ksize=3)
        + variance_of_laplacian(reconstructed, ksize=5)
        + variance_of_laplacian(reconstructed, ksize=7)
    )

    # Compute total loss and return
    loss = (
        alpha * kl_loss
        + beta * discrimination_reconstruction_loss
        + gamma * discrimination_generation_loss
        + delta * sharpness_loss_generated
        + epsilon * sharpness_loss_reconstructed
    )
    return (
        loss,
        kl_loss,
        discrimination_reconstruction_loss,
        discrimination_generation_loss,
        sharpness_loss_generated,
        sharpness_loss_reconstructed,
        reconstructed,
        generated,
    )


def compute_discriminator_loss(
    vae: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    reconstructed: tf.Tensor,
    generated: tf.Tensor,
    beta: float = 1e0,
    gamma: float = 1e0,
) -> Tuple[float, float, float, float]:
    """
    Compute loss for training discriminator network

    Args:
        vae: VAE model - should have encode(), decode(), reparameterize(), and sample()
             methods
        discriminator: Discriminator model
        images: Training images
        labels: Training labels
        reconstructed: Reconstructions of training images by autoencoder
        generated: Fake images generated by autoencoder
        beta: Contribution of loss on interpolated images to total loss
        gamma: Contribution of loss on generated images to total loss

    Returns:
        Total loss
        Loss on real images
        Loss on reconstructed images
        Loss on generated images
    """
    # Generate 4 blocks from scratch, and interpolate the centre (like in flood generation)
    minibatch_size, _, block_width, context_channels = images.shape
    z_interp = tf.random.normal(shape=[minibatch_size * 4, vae.latent_dim_])
    generated = vae.sample(z_interp)
    context = tf.reshape(
        generated, [minibatch_size, block_width, block_width, context_channels]
    )
    interpolated = vae.call(context)

    # Run real images, reconstructed images, and fake images through discriminator
    real_output = discriminator(labels)
    interpolated_output = discriminator(interpolated)
    generated_output = discriminator(generated)

    # Compute loss components
    real_loss = bce(tf.ones_like(real_output), real_output)
    interpolated_loss = bce(tf.zeros_like(interpolated_output), interpolated_output)
    generated_loss = bce(tf.zeros_like(generated_output), generated_output)

    # Compute total loss and return
    total_loss = real_loss + beta * interpolated_loss + gamma * generated_loss
    return total_loss, real_loss, interpolated_loss, generated_loss


@tf.function
def train_step(
    images: tf.Tensor,
    labels: tf.Tensor,
    autoencoder: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    autoencoder_optimizer: "Optimizer",
    discriminator_optimizer: "Optimizer",
    loss_metrics: Dict[str, "Metrics"],
    should_train_vae: bool,
    should_train_discriminator: bool,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        autoencoder: Autoencoder model
        discriminator: Discriminator model
        autoencoder_optimizer: Optimizer for autoencoder model
        discriminator_optimizer: Optimizer for discriminator model
        loss_metrics: Dictionary of metrics to log
        batch_size: Number of training examples in a batch
        latent_dim: Size of latent space
        should_train_vae: Update vae weights if True, leave static otherwise
        should_train_discriminator: Update discriminator weights if True, leave static otherwise
    """
    # Compute losses
    with tf.GradientTape() as vae_tape, tf.GradientTape() as disc_tape:
        (
            vae_loss,
            kl_loss,
            discrimination_reconstruction_loss,
            discrimination_generation_loss,
            generation_sharpness_loss,
            reconstruction_sharpness_loss,
            reconstructed_images,
            generated_images,
        ) = compute_vae_loss(
            autoencoder,
            discriminator,
            images,
            labels,
        )
        (
            discriminator_loss,
            discriminator_real_loss,
            discriminator_interepolated_loss,
            discriminator_generated_loss,
        ) = compute_discriminator_loss(
            vae=autoencoder,
            discriminator=discriminator,
            images=images,
            labels=labels,
            reconstructed=reconstructed_images,
            generated=generated_images,
        )

    # Compute gradients from losses
    vae_gradients = vae_tape.gradient(vae_loss, autoencoder.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        discriminator_loss, discriminator.trainable_variables
    )

    # Update weights if desired
    if should_train_vae:
        autoencoder_optimizer.apply_gradients(
            zip(vae_gradients, autoencoder.trainable_variables)
        )
    if should_train_discriminator:
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

    # Log losses to their respective metrics
    loss_metrics["vae_loss_metric"](vae_loss)
    loss_metrics["kl_loss_metric"](kl_loss)
    loss_metrics["generation_sharpness_loss_metric"](generation_sharpness_loss)
    loss_metrics["reconstruction_sharpness_loss_metric"](reconstruction_sharpness_loss)
    loss_metrics["discrimination_reconstruction_loss_metric"](
        discrimination_reconstruction_loss
    )
    loss_metrics["discrimination_generation_loss_metric"](
        discrimination_generation_loss
    )
    loss_metrics["discriminator_loss_metric"](discriminator_loss)
    loss_metrics["discriminator_real_loss_metric"](discriminator_real_loss)
    loss_metrics["discriminator_interepolated_loss_metric"](
        discriminator_interepolated_loss
    )
    loss_metrics["discriminator_generated_loss_metric"](discriminator_generated_loss)


def compute_mse_losses(
    autoencoder: tf.keras.models.Model,
    train_images: tf.Tensor,
    train_labels: tf.Tensor,
    val_images: tf.Tensor,
    val_labels: tf.Tensor,
) -> Tuple[float, float]:
    """
    Compute loss on training and validation data.

    Args:
        autoencoder: Autoencoder model
        train_images: Batch of training data
        val_images: Batch of validation data
    """
    reproduced_train_images = autoencoder.call(train_images)
    reproduced_val_images = autoencoder.call(val_images)
    train_loss = mse(train_labels, reproduced_train_images)
    val_loss = mse(val_labels, reproduced_val_images)
    return train_loss, val_loss


def stanford_dogs_preprocess(row: List[tf.Tensor]) -> tf.Tensor:
    """
    Preprocess a row from the stanford dogs dataset into the desired format.
    """
    floating_point_image = tf.image.convert_image_dtype(row["image"], dtype=tf.float32)
    return tf.image.resize(
        floating_point_image,
        (64, 64),
        method="nearest",
    )


class LRS(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule

    Currently a piecewise defined function that is initially static at the maximum
    learning rate, then ramps down until it reaches the minimum learning rate, where it
    stays.

    i.e. it looks something like this:
    _____
         \
          \
           \
            \_____
    """

    def __init__(
        self,
        max_lr: float = 1e-4,
        min_lr: float = 5e-6,
        start_decay_epoch: int = 30,
        stop_decay_epoch: int = 800,
        steps_per_epoch: int = 390,  # Cats dataset
    ) -> "LRS":
        """
        Initialize learning rate schedule

        Args:
            step: Which training step we are up to
            max_lr: Initial value of lr
            min_lr: Final value of lr
            start_decay_epoch: Stay at max_lr until this many epochs have passed, then start decaying
            stop_decay_epoch: Reach min_lr after this many epochs have passed, and stop decaying

        Returns:
            Learning rate schedule object
        """
        self.max_lr_ = max_lr
        self.min_lr_ = min_lr
        self.start_decay_epoch_ = start_decay_epoch
        self.stop_decay_epoch_ = stop_decay_epoch
        self.steps_per_epoch_ = steps_per_epoch

    @tf.function
    def __call__(self, step: tf.Tensor) -> float:
        """
        Compute the learning rate for the given step

        Args:
            step: Which training step we are up to

        Returns:
            Learning rate for this step
        """
        # Initial flat LR period, before ramping down
        if step < self.start_decay_epoch_ * self.steps_per_epoch_:
            return self.max_lr_

        # Final flat LR period, after ramping down
        if step > self.stop_decay_epoch_ * self.steps_per_epoch_:
            return self.min_lr_

        # Ramping period
        gradient = -(self.max_lr_ - self.min_lr_) / (
            (self.stop_decay_epoch_ - self.start_decay_epoch_) * self.steps_per_epoch_
        )
        return self.max_lr_ + gradient * (
            step - self.start_decay_epoch_ * self.steps_per_epoch_
        )


def train(
    autoencoder: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    autoencoder_optimizer: "Optimizer",
    discriminator_optimizer: "Optimizer",
    model_dir: str,
    checkpoint: tf.train.Checkpoint,
    checkpoint_prefix: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (256, 256, 3),
    buffer_size: int = 1000,
    batch_size: int = 128,
    epochs_per_turn: int = 1,
    latent_dim: int = 10,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Train the model.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        generator_optimizer: Generator optimizer
        discriminator_optimizer: Discriminator optimizer
        model_dir: Working dir for this experiment
        checkpoint: Checkpoint to save to
        checkpoint_prefix: Prefix to checkpoint numbers in checkpoint filenames
        dataset_path: Path to directory tree containing training data
        val_path: Path to directory tree containing data to be used for validation
        epochs: How many full passes through the dataset to make
        train_crop_shape: Desired shape of training crops from full images
        buffer_size: Number of images to randomly sample from at a time
        batch_size: Number of training examples in a batch
        epochs_per_turn: How long to train one model before switching to the other
        latent_dim: Number of noise inputs to generator
        num_examples_to_generate: How many examples to generate on each epoch
        continue_from_checkpoint: Restore weights from checkpoint file if given, start
                                  from scratch otherwise.
    """
    if not debug:
        # Die if there are uncommitted changes in the repo
        if modified_files_in_git_repo():
            return

        # Write commit hash to model directory
        os.makedirs(model_dir, exist_ok=True)
        write_commit_hash_to_model_dir(model_dir)

    # Shape of 3x3 blocks context region
    rows, cols, channels = train_crop_shape
    context_shape = (rows, cols, channels * 4)

    # Pixel Art dataset
    train_images = (
        tf.data.Dataset.from_generator(
            PixelArtFloodDataset(
                dataset_path=dataset_path, crop_shape=train_crop_shape, flip_x=False
            ),
            output_signature=(
                tf.TensorSpec(shape=context_shape, dtype=tf.float32),
                tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
            ),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_images = (
        tf.data.Dataset.from_generator(
            PixelArtFloodDataset(
                dataset_path=val_path, crop_shape=train_crop_shape, flip_x=False
            ),
            output_signature=(
                tf.TensorSpec(shape=context_shape, dtype=tf.float32),
                tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
            ),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    # # Stanford dogs dataset
    # dataset = tfds.load(name="stanford_dogs")
    # train_images = (
    #     dataset["train"]
    #     .map(stanford_dogs_preprocess)
    #     # .map(lambda x: data_augmentation(x, training=True))
    #     .shuffle(batch_size)
    #     .batch(batch_size)
    #     .cache()
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # val_images = (
    #     dataset["test"]
    #     .map(stanford_dogs_preprocess)
    #     .shuffle(batch_size)
    #     .batch(batch_size)
    #     .cache()
    #     .prefetch(tf.data.AUTOTUNE)
    # )

    # # CelebA faces dataset
    # dataset = tfds.load(name="celeb_a")
    # train_images = (
    #     dataset["train"]
    #     .map(stanford_dogs_preprocess)
    #     # .map(lambda x: data_augmentation(x, training=True))
    #     .shuffle(batch_size)
    #     .batch(batch_size)
    #     .cache()
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # val_images = (
    #     dataset["test"]
    #     .map(stanford_dogs_preprocess)
    #     .shuffle(batch_size)
    #     .batch(batch_size)
    #     .cache()
    #     .prefetch(tf.data.AUTOTUNE)
    # )

    # # Cats and dogs dataset
    # train_images = (
    #     tf.data.Dataset.from_generator(
    #         ResizeDataset(dataset_path=dataset_path, output_shape=train_crop_shape),
    #         output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
    #     )
    #     .shuffle(buffer_size)
    #     .batch(batch_size)
    #     .cache()
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # val_images = (
    #     tf.data.Dataset.from_generator(
    #         ResizeDataset(dataset_path=val_path, output_shape=train_crop_shape),
    #         output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
    #     )
    #     .shuffle(buffer_size)
    #     .batch(batch_size)
    # )

    # Save a few images for visualisation
    train_test_image_batch, train_test_label_batch = next(iter(train_images))
    train_test_images = train_test_image_batch[:8, ...]
    val_test_image_batch, val_test_label_batch = next(iter(val_images))
    val_test_images = val_test_image_batch[:8, ...]

    # Make progress dir and reproductions dir for outputs
    os.makedirs(progress_dir := os.path.join(model_dir, "progress"), exist_ok=True)
    os.makedirs(
        reproductions_dir := os.path.join(model_dir, "reproductions"), exist_ok=True
    )
    os.makedirs(
        val_reproductions_dir := os.path.join(model_dir, "val_reproductions"),
        exist_ok=True,
    )
    os.makedirs(
        flood_generations_dir := os.path.join(model_dir, "flood_generations"),
        exist_ok=True,
    )

    # Set starting and end epoch according to whether we are continuing training
    epoch_log_file = os.path.join(model_dir, "epoch_log")
    if continue_from_checkpoint is not None:
        epoch_start = 15 * int(
            continue_from_checkpoint.strip()[continue_from_checkpoint.rfind("-") + 1 :]
        )
    else:
        epoch_start = 0
    epoch_stop = epoch_start + epochs

    # Define metrics
    vae_loss_metric = tf.keras.metrics.Mean("vae_loss", dtype=tf.float32)
    kl_loss_metric = tf.keras.metrics.Mean("kl_loss", dtype=tf.float32)
    generation_sharpness_loss_metric = tf.keras.metrics.Mean(
        "generation_sharpness_loss", dtype=tf.float32
    )
    reconstruction_sharpness_loss_metric = tf.keras.metrics.Mean(
        "reconstruction_sharpness_loss", dtype=tf.float32
    )
    discrimination_reconstruction_loss_metric = tf.keras.metrics.Mean(
        "discrimination_reconstruction_loss", dtype=tf.float32
    )
    discrimination_generation_loss_metric = tf.keras.metrics.Mean(
        "discrimination_generation_loss", dtype=tf.float32
    )
    discriminator_loss_metric = tf.keras.metrics.Mean(
        "discriminator_loss", dtype=tf.float32
    )
    discriminator_real_loss_metric = tf.keras.metrics.Mean(
        "discriminator_real_loss", dtype=tf.float32
    )
    discriminator_interepolated_loss_metric = tf.keras.metrics.Mean(
        "discriminator_interepolated_loss", dtype=tf.float32
    )
    discriminator_generated_loss_metric = tf.keras.metrics.Mean(
        "discriminator_generated_loss", dtype=tf.float32
    )

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(model_dir, "logs", "gradient_tape", current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Use the same seed throughout training, to see what the model does with the same input as it trains.
    seed = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # Start by training both networks
    should_train_vae = True
    should_train_discriminator = True

    for epoch in range(epoch_start, epoch_stop):
        start = time.time()

        for step, (image_batch, label_batch) in enumerate(train_images):
            # Perform training step
            train_step(
                images=image_batch,
                labels=label_batch,
                autoencoder=autoencoder,
                discriminator=discriminator,
                autoencoder_optimizer=autoencoder_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                loss_metrics={
                    "vae_loss_metric": vae_loss_metric,
                    "kl_loss_metric": kl_loss_metric,
                    "generation_sharpness_loss_metric": generation_sharpness_loss_metric,
                    "reconstruction_sharpness_loss_metric": reconstruction_sharpness_loss_metric,
                    "discrimination_reconstruction_loss_metric": discrimination_reconstruction_loss_metric,
                    "discrimination_generation_loss_metric": discrimination_generation_loss_metric,
                    "discriminator_loss_metric": discriminator_loss_metric,
                    "discriminator_real_loss_metric": discriminator_real_loss_metric,
                    "discriminator_interepolated_loss_metric": discriminator_interepolated_loss_metric,
                    "discriminator_generated_loss_metric": discriminator_generated_loss_metric,
                },
                should_train_vae=should_train_vae,
                should_train_discriminator=should_train_discriminator,
            )

            if step % 5 == 0:
                print(
                    UPDATE_TEMPLATE.format(
                        epoch=epoch + 1,
                        step=step,
                        epoch_time=time.time() - start,
                        vae_loss=vae_loss_metric.result(),
                        kl_loss=kl_loss_metric.result(),
                        generation_sharpness_loss=generation_sharpness_loss_metric.result(),
                        reconstruction_sharpness_loss=reconstruction_sharpness_loss_metric.result(),
                        discriminator_loss=discriminator_loss_metric.result(),
                    )
                )

        # Produce demo output every epoch the generator trains
        generated = generate_and_save_images(autoencoder, epoch + 1, seed, progress_dir)

        # Flood generate a bigger image
        flood_generated_image = flood_generate(autoencoder, seed)
        save_flood_generated_image(
            flood_generated_image,
            output_dir=flood_generations_dir,
            epoch=epoch,
        )

        # Show some examples of how the model reconstructs its inputs.
        train_reconstructed = show_reproduction_quality(
            autoencoder,
            epoch + 1,
            train_test_images,
            train_test_label_batch,
            reproductions_dir,
        )
        val_reconstructed = show_reproduction_quality(
            autoencoder,
            epoch + 1,
            val_test_images,
            val_test_label_batch,
            val_reproductions_dir,
        )

        # Compute reconstruction loss on training and validation data
        train_loss, val_loss = compute_mse_losses(
            autoencoder,
            train_test_image_batch,
            train_test_label_batch,
            val_test_image_batch,
            val_test_label_batch,
        )

        # Get sample of encoder output
        z_mean, z_log_var = autoencoder.encode(train_test_image_batch)
        encoder_output_train = autoencoder.reparameterize(z_mean, z_log_var)
        z_mean, z_log_var = autoencoder.encode(val_test_image_batch)
        encoder_output_val = autoencoder.reparameterize(z_mean, z_log_var)

        # Write to logs
        with summary_writer.as_default():
            # Learning rates
            tf.summary.scalar(
                "VAE learning rate",
                # autoencoder_optimizer.learning_rate,
                autoencoder_optimizer.learning_rate(epoch * step),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator learning rate",
                # discriminator_optimizer.learning_rate,
                discriminator_optimizer.learning_rate(epoch * step),
                step=epoch,
            )

            # Encoder outputs
            tf.summary.histogram(
                "encoder output train", encoder_output_train, step=epoch
            )
            tf.summary.histogram("encoder output val", encoder_output_val, step=epoch)
            tf.summary.histogram(
                "random normal noise",
                tf.random.normal(shape=[num_examples_to_generate, latent_dim]),
                step=epoch,
            )
            tf.summary.scalar(
                "encoder output train: mean",
                np.mean(encoder_output_train.numpy()),
                step=epoch,
            )
            tf.summary.scalar(
                "encoder output train: stddev",
                np.std(encoder_output_train.numpy()),
                step=epoch,
            )
            tf.summary.scalar(
                "encoder output val: mean",
                np.mean(encoder_output_val.numpy()),
                step=epoch,
            )
            tf.summary.scalar(
                "encoder output val: stddev",
                np.std(encoder_output_val.numpy()),
                step=epoch,
            )

            # Losses
            tf.summary.scalar("VAE loss metric", vae_loss_metric.result(), step=epoch)
            tf.summary.scalar("KL loss metric", kl_loss_metric.result(), step=epoch)
            tf.summary.scalar(
                "Generation sharpness loss metric",
                generation_sharpness_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "Reconstruction sharpness loss metric",
                reconstruction_sharpness_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "Discrimination generation loss metric",
                discrimination_generation_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "Discrimination reconstruction loss metric",
                discrimination_reconstruction_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator loss metric",
                discriminator_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator real loss metric",
                discriminator_real_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator reconstructed loss metric",
                discriminator_interepolated_loss_metric.result(),
                step=epoch,
            )
            tf.summary.scalar(
                "discriminator generated loss metric",
                discriminator_generated_loss_metric.result(),
                step=epoch,
            )

            # MSE reconstruction losses
            tf.summary.scalar("MSE train loss", train_loss, step=epoch)
            tf.summary.scalar("MSE validation loss", val_loss, step=epoch)

            # Example outputs
            tf.summary.image(
                "reconstructed train images", train_reconstructed, step=epoch
            )
            tf.summary.image("reconstructed val images", val_reconstructed, step=epoch)
            tf.summary.image("generated images", generated, step=epoch)
            tf.summary.image(
                "flood generated image",
                tf.expand_dims(flood_generated_image, axis=0),
                step=epoch,
            )

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Also close all pyplot figures. It is expensive to do this every epoch
            plt.close("all")

            # Write our last epoch down in case we want to continue
            with open(epoch_log_file, "w", encoding="utf-8") as epoch_log:
                epoch_log.write(str(epoch))

        # Switch who trains if appropriate
        if should_train_discriminator == should_train_vae:
            should_train_vae = False
            should_train_discriminator = True
        elif should_train_discriminator and discriminator_loss_metric.result() > 0.1:
            print("Not switching who trains, discriminator fooled too easily")
        elif should_train_vae and discriminator_loss_metric.result() < 1.0:
            print("Not switching who trains, unable to fool discriminator")
        else:
            should_train_vae = not should_train_vae
            should_train_discriminator = not should_train_discriminator
            print(
                f"Switching who trains: {should_train_vae=}, {should_train_discriminator=}"
            )

        # Reset metrics every epoch
        vae_loss_metric.reset_states()
        kl_loss_metric.reset_states()
        generation_sharpness_loss_metric.reset_states()
        reconstruction_sharpness_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()
        discrimination_reconstruction_loss_metric.reset_states()
        discrimination_generation_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()
        discriminator_real_loss_metric.reset_states()
        discriminator_interepolated_loss_metric.reset_states()
        discriminator_generated_loss_metric.reset_states()


def generate(
    autoencoder: tf.keras.models.Model,
    decoder_input: Optional[str] = None,
    latent_dim: int = 50,
    save_output: bool = False,
    sample: bool = True,
    flood_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Generate some pixel art

    Args:
        generator: Generator model
        generator_input: Path to a 10x10 grayscale image to use as input. Random noise
                         used if not given.
        save_output: Save images instead of displaying
        sample: Apply sigmoid to decoder output if true, return logits othverwise
        flood_shape: If given, flood generate an image of given shape (in blocks)
                     instead of generating a single block from noise.
    """
    if decoder_input is not None:
        input_raw = tf.io.read_file(decoder_input)
        input_decoded = tf.image.decode_image(input_raw)
        latent_input = tf.reshape(input_decoded, [200, latent_dim])
    else:
        latent_input = tf.random.normal([200, latent_dim])
    i = 0
    while True:
        if flood_shape is not None:
            output = tf.expand_dims(
                flood_generate(autoencoder, seed=latent_input, shape=flood_shape),
                axis=0,
            )
        elif sample:
            output = autoencoder.sample(latent_input)
        else:
            output = autoencoder.decoder_(latent_input, training=False)
        generated_rgb_image = np.array((output[0, :, :, :] * 255.0)).astype(np.uint8)
        # generated_rgb_image = cv2.cvtColor(generated_hsv_image, cv2.COLOR_HSV2RGB)
        plt.close("all")
        plt.imshow(generated_rgb_image)
        plt.axis("off")
        if save_output:
            plt.savefig(f"generated_{i}.png")
            print(f"image generated to: generated_{i}.png")
            input("press enter to generate another image")
            i += 1
        else:
            plt.show()
        latent_input = tf.random.normal([200, latent_dim])


def regenerate_images(slider_value: Optional[float] = None) -> None:
    """
    Render input as 10x10 grayscale image, feed it into the generator to generate a new
    output, and update the GUI.

    Args:
        slider_value: Position of slider, between 0 and 1. If not given, randomize input.
    """
    global generator_
    global generated_image
    global latent_input
    global input_neuron
    global latent_input_canvas
    global generator_output_canvas
    if slider_value is not None:
        latent_input[input_neuron] = slider_value
    else:
        latent_input = tf.random.normal([1, 100])
    # breakpoint()
    generated_output = np.array(
        generator_(latent_input, training=False)[0, :, :, :] * 255.0
    ).astype(int)
    print(generated_output.shape)
    print(latent_input.shape)
    generated_image = PIL.ImageTk.PhotoImage(
        PIL.Image.fromarray(generated_output, mode="RGB")
    )
    latent_image = PIL.ImageTk.PhotoImage(
        PIL.Image.fromarray(np.array(tf.reshape(latent_input, [10, 10])))
    )
    latent_input_canvas.create_image(0, 0, image=latent_image, anchor=tkinter.NW)
    generator_output_canvas.create_image(0, 0, image=generated_image, anchor=tkinter.NW)


def view_latent_space(generator: tf.keras.Sequential) -> None:
    """
    View and modify the latent input using a basic Tkinter GUI.

    Args:
        generator: Generator model
    """
    # Create the window
    window = tkinter.Tk()

    # Add canvases to display images
    global latent_input_canvas
    global generator_output_canvas
    latent_input_canvas = tkinter.Canvas(master=window, width=100, height=100)
    generator_output_canvas = tkinter.Canvas(master=window, width=128, height=128)

    # Slider to set input activation for selected noise input
    input_value_slider = tkinter.Scale(
        master=window,
        from_=0.0,
        to=255.0,
        command=lambda: regenerate_images(slider_value=input_value_slider.get()),
        orient=tkinter.HORIZONTAL,
        length=600,
        width=30,
        bg="blue",
    )

    # Buttons
    randomize_button = tkinter.Button(
        master=window,
        text="randomize",
        command=regenerate_images,
        padx=40,
        pady=20,
        bg="orange",
    )

    # Set layout
    generator_output_canvas.grid(row=0, column=0)
    latent_input_canvas.grid(row=1, column=0)
    input_value_slider.grid(row=2, column=0)
    randomize_button.grid(row=3, column=0)

    # Define the globals
    global generator_
    generator_ = generator

    # Start randomly
    regenerate_images()

    # Run gui
    window.mainloop()


def main(
    mode: str,
    model_dir: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (128, 128, 3),
    buffer_size: int = 1000,
    batch_size: int = 128,
    epochs_per_turn: int = 1,
    latent_dim: int = 256,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
    decoder_input: Optional[str] = None,
    save_generator_output: bool = False,
    flood_generate_shape: Optional[Tuple[int, int]] = None,
    debug: bool = False,
    sample: bool = True,
) -> None:
    """
    Main routine.

    Args:
        mode: One of "train" "generate" "discriminate"
        model_dir: Working dir for this experiment
        dataset_path: Path to directory tree containing training data
        val_path: Path to directory tree containing data to be used for validation
        epochs: How many full passes through the dataset to make
        train_crop_shape: Desired shape of training crops from full images
        buffer_size: Number of images to randomly sample from at a time
        batch_size: Number of training examples in a batch
        epochs_per_turn: How long to train one model before switching to the other
        latent_dim: Number of noise inputs to generator
        num_examples_to_generate: How many examples to generate on each epoch
        continue_from_checkpoint: Restore weights from checkpoint file if given, start
                                  from scratch otherwise.
        decoder_input: Path to a 10x10 grayscale image, to be used as input to the
                         generator. Noise used if None
        save_generator_output: Save generated images instead of displaying
    """
    # STEPS_PER_EPOCH = 225  # pixel_art - minibatch size 128
    # STEPS_PER_EPOCH = 600  # expanded pixel_art - minibatch size 128
    # STEPS_PER_EPOCH = 665  # expanded pixel_art - minibatch size 128 - checkerboard
    # STEPS_PER_EPOCH = (
    #     330  # expanded pixel_art - minibatch size 128 - checkerboard - no flipping
    # )
    STEPS_PER_EPOCH = 83  # expanded pixel_art - 128 crops - minibatch size 128 - checkerboard - no flipping

    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=[STEPS_PER_EPOCH * epoch for epoch in (30, 200)],
    #     values=[1e-4, 7e-5, 3e-5],
    #     name=None,
    # )
    autoencoder_lr = LRS(
        max_lr=1e-4,
        min_lr=1e-5,
        start_decay_epoch=50,
        stop_decay_epoch=1500,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    discriminator_lr = LRS(
        max_lr=1e-4,
        min_lr=3e-6,
        start_decay_epoch=50,
        stop_decay_epoch=1500,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    # lr = tfa.optimizers.CyclicalLearningRate(
    #     initial_learning_rate=1e-5,
    #     maximal_learning_rate=3e-4,
    #     scale_fn=lambda x: 1 / (1.1 ** (x - 1)),
    #     step_size=3 * STEPS_PER_EPOCH,
    # )

    autoencoder = VAE(latent_dim=latent_dim, input_shape=train_crop_shape[:2] + (12,))
    discriminator = discriminator_model(input_shape=train_crop_shape)
    # autoencoder_optimizer = tf.keras.optimizers.Adam(lr)
    # discriminator_optimizer = tf.keras.optimizers.Adam(lr)
    autoencoder_optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-7, learning_rate=autoencoder_lr
    )
    discriminator_optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-7, learning_rate=discriminator_lr
    )
    # optimizer = tf.keras.optimizers.Adam(clr)
    # step = tf.Variable(0, trainable=False)
    # optimizer = tfa.optimizers.AdamW(
    #     weight_decay=clr, learning_rate=clr
    # )

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        autoencoder=autoencoder,
        discriminator=discriminator,
        autoencoder_optimizer=autoencoder_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )

    # Restore model from checkpoint
    if continue_from_checkpoint is not None:
        checkpoint.restore(continue_from_checkpoint)

    if mode == "train":
        train(
            autoencoder=autoencoder,
            discriminator=discriminator,
            autoencoder_optimizer=autoencoder_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            model_dir=model_dir,
            checkpoint=checkpoint,
            checkpoint_prefix=checkpoint_prefix,
            dataset_path=dataset_path,
            val_path=val_path,
            epochs=epochs,
            train_crop_shape=train_crop_shape,
            buffer_size=buffer_size,
            batch_size=batch_size,
            epochs_per_turn=epochs_per_turn,
            latent_dim=latent_dim,
            num_examples_to_generate=num_examples_to_generate,
            continue_from_checkpoint=continue_from_checkpoint,
            debug=debug,
        )
    elif mode == "generate":
        generate(
            autoencoder=autoencoder,
            decoder_input=decoder_input,
            flood_shape=flood_generate_shape,
            latent_dim=latent_dim,
            save_output=save_generator_output,
            sample=sample,
        )
    elif mode == "view-latent-space":
        view_latent_space(generator=autoencoder.decoder_)


def get_args() -> argparse.Namespace:
    """
    Define and parse command line arguments

    Returns:
        Argument values as argparse namespace
    """
    parser = argparse.ArgumentParser(
        "Generate pixel art", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "mode",
        type=str,
        help=f"Mode of operation, must be one of {MODES_OF_OPERATION}",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="Generate, discriminate, or continue training model from this checkpoint",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="/mnt/storage/ml/data/pixel-art/train",
        help="Path to dataset directory, containing training images",
    )
    parser.add_argument(
        "--debug",
        "-e",
        action="store_true",
        help="Don't check git status, current test is just for debugging",
    )
    parser.add_argument(
        "--flood-generate-shape",
        "-f",
        type=str,
        help="Dimensions (in blocks) of flood generated image, e.g. '4,4'",
    )
    parser.add_argument(
        "--generator-input",
        "-g",
        type=str,
        help="10x10 grayscale image, flattened and used as input to the generator. "
        "Random noise is used if not given",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--no-sample",
        "-n",
        action="store_true",
        help="Directly feed noise to decoder instead of running sample",
    )
    parser.add_argument(
        "--save-output",
        "-s",
        action="store_true",
        help="Save generator output to file instead of displaying",
    )
    parser.add_argument(
        "--validation-dataset",
        "-v",
        type=str,
        default="/mnt/storage/ml/data/pixel-art/val",
        help="Path to dataset directory, containing images to test with",
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
    if args.mode not in MODES_OF_OPERATION:
        print(f"Invalid mode: {args.mode}, must be one of {MODES_OF_OPERATION}")
        return 1
    main(
        mode=args.mode,
        model_dir=os.path.join(
            os.path.expanduser("/mnt/storage/ml/models"), args.model_name
        ),
        dataset_path=args.dataset,
        debug=args.debug,
        flood_generate_shape=tuple(
            int(dim) for dim in args.flood_generate_shape.strip().split(",")
        )
        if args.flood_generate_shape is not None
        else None,
        val_path=args.validation_dataset,
        continue_from_checkpoint=args.checkpoint,
        decoder_input=args.generator_input,
        save_generator_output=args.save_output,
        sample=not args.no_sample,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
