#!/usr/bin/env python3


"""
Generative adversarial network to generate pixel art wallpapers
"""


__author__ = "Hamish Morgan"


from typing import Any, Iterable, Dict, Iterator, List, Optional, Tuple

import argparse
import datetime
import math
import os
import random
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


from hml.architectures.convolutional.generators.sample_gan_generator import (
    model as generator_model,
    GENERATOR_LATENT_DIM,
)

from hml.architectures.convolutional.discriminators.sample_gan_discriminator import (
    model as discriminator_model,
)

from hml.data_pipelines.unsupervised.pixel_art_sigmoid import PixelArtSigmoidDataset

from hml.util.git import modified_files_in_git_repo, write_commit_hash_to_model_dir


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")

UPDATE_TEMPLATE = """
Epoch: {epoch}    Step: {step}    Time: {epoch_time:.2f}

Generator loss:     {generator_loss:>6.4f}
Discriminator loss: {discriminator_loss:>6.4f}
"""


# This method returns a helper function to compute mean squared error loss
bce = tf.keras.losses.BinaryCrossentropy()


global generator_
global generated_image
global latent_input
global input_neuron
global latent_input_canvas
global generator_output_canvas


def save_tensor_image(image: tf.Tensor, path: str) -> None:
    """
    Save a tensor to disk as an image

    Args:
        image: Image as tf Tensor
        path: Path to save image to
    """
    PIL.Image.fromarray((image.numpy() * 255.0).astype(np.uint8)[0, ...]).save(path)


def generate_and_save_images(
    generator: tf.keras.Sequential,
    epoch: int,
    consistent_input: tf.Tensor,
    consistent_generations_dir: str,
    random_generations_dir: str,
) -> tf.Tensor:
    """
    Generate images from the generator to show training progress

    We will use one seed consistently every epoch to show the model training, but also
    generate a new random seed on every epoch to better gauge the variance in the
    generator's output.

    Args:
        generator: Generator model
        epoch: Current epoch (for save paths)
        consistent_input: Seed for first generated image
        model_dir: Root model directory

    Returns:
        Image generated from consistent seed
        Image generated from random seed
    """
    # Generate image from consistent seed
    consistent_generation = generator(consistent_input)

    # Generate new random seed and use it to generate a new image
    random_input = tf.random.normal(tf.shape(consistent_input))
    random_generation = generator(random_input)

    # Construct save paths
    consistent_generation_path = os.path.join(
        consistent_generations_dir, f"image_at_epoch_{epoch:04d}.png"
    )
    random_generation_path = os.path.join(
        random_generations_dir, f"image_at_epoch_{epoch:04d}.png"
    )

    # Write to filesystem
    save_tensor_image(image=consistent_generation, path=consistent_generation_path)
    save_tensor_image(image=random_generation, path=random_generation_path)

    # Stack generated images and return
    return consistent_generation, random_generation


def sorted_locals(locals_: Dict[str, Any]) -> Iterator[Tuple[int, int]]:
    """
    Sort local variables that don't start with underscores by their size

    Params:
        locals_: Return value of locals() in target environment

    Returns:
        List of (variable name, size in bytes) sorted by size
    """
    return sorted(
        [
            (key, sys.getsizeof(value))
            for key, value in locals_.items()
            if not key.startswith("_")
        ],
        key=lambda elem: elem[1],
    )


def sample_minibatch(
    fullsize_generated_images: tf.Tensor,
    minibatch_shape: tf.TensorShape,
    num_fullsize_generations: int,
) -> tf.Tensor:
    """
    From a spatially large image, randomly sample smaller tiles of a given shape

    This helps to allow training a big generator with a small discriminator

    Args:
        images: Large images as tensors (Num images x X x Y x Channels)
        minibatch_shape: Required output shape
        num_fullsize_generations: Number of full-sized images were generated

    Returns:
        Tiles randomly sampled from large images
    """
    # Extract relevant dimensions
    num_images, image_height, image_width, _ = tf.shape(fullsize_generated_images)
    minibatch_size, tile_size, _, _ = minibatch_shape

    # Sample tiles randomly from full size image
    tile_max_y = image_height - tile_size
    tile_max_x = image_width - tile_size
    tile_ys = np.random.uniform(low=0, high=tile_max_y, size=minibatch_size).astype(int)
    tile_xs = np.random.uniform(low=0, high=tile_max_x, size=minibatch_size).astype(int)
    tile_source_images = np.random.uniform(
        low=0, high=num_images, size=minibatch_size
    ).astype(int)

    # Make shuffled list of tiles
    tiles = [
        fullsize_generated_images[src_img, y : y + tile_size, x : x + tile_size, :]
        # for src_img, y, x in zip(tile_source_images, tile_ys, tile_xs)
        for src_img in range(num_images)
        for y in range(0, tile_max_y, tile_size)
        for x in range(0, tile_max_x, tile_size)
    ]
    random.shuffle(tiles)

    # Return first minibatch of shuffled tiles, stacked
    return tf.stack(
        values=tiles[:minibatch_size],
        axis=0,
    )


def compute_generator_loss(generated_output: tf.Tensor) -> float:
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


# @tf.function
def train_step(
    images: tf.Tensor,
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    generator_optimizer: "Optimizer",
    discriminator_optimizer: "Optimizer",
    loss_metrics: Dict[str, "Metrics"],
    should_train_generator: bool,
    should_train_discriminator: bool,
    num_fullsize_generations: int = 1,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        generator: Generator model
        discriminator: Discriminator model
        generator_optimizer: Optimizer for generator model
        discriminator_optimizer: Optimizer for discriminator model
        loss_metrics: Dictionary of metrics to log
        should_train_generator: Update generator weights if True, leave static otherwise
        should_train_discriminator: Update discriminator weights if True, leave static
                                    otherwise
    """
    # Generate a new seed to generate a set of new images
    noise = tf.random.normal([num_fullsize_generations, GENERATOR_LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generate images
        generated_images = generator(noise, training=True)

        # Get a random sample of blocks from generated images
        generated_minibatch = sample_minibatch(
            generated_images,
            minibatch_shape=tf.shape(images),
            num_fullsize_generations=num_fullsize_generations,
        )

        # Pass real and fake images through discriminator
        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_minibatch, training=True)

        # Compute losses
        discriminator_loss = compute_discriminator_loss(real_output, generated_output)
        generator_loss = compute_generator_loss(generated_output)

    # Compute gradients from losses
    generator_gradients = gen_tape.gradient(
        generator_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        discriminator_loss, discriminator.trainable_variables
    )

    # Update weights if desired
    if should_train_generator:
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
    if should_train_discriminator:
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

    # Log losses to their respective metrics
    loss_metrics["generator_loss"](generator_loss)
    loss_metrics["discriminator_loss"](discriminator_loss)


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


def decide_who_trains(
    should_train_generator: bool,
    should_train_discriminator: bool,
    this_generator_loss: float,
    last_generator_loss: float,
    switch_training_loss_delta: float = 0.1,
    start_training_generator_loss_threshold: float = 1.0,
    start_training_discriminator_loss_threshold: float = 0.1,
) -> Tuple[bool, bool]:
    """
    Decide which network should train and which should be frozen

    This is determined based on both an absolute loss threshold and a delta loss
    threshold. If after training for an epoch, the generator loss has passed the
    relevant absolute threshold, or the generator loss has not changed significantly, we
    switch to training the other network.
    """
    loss_delta = abs(this_generator_loss - last_generator_loss)
    if should_train_discriminator == should_train_generator:
        should_train_generator = False
        should_train_discriminator = True
    elif (
        should_train_discriminator
        and this_generator_loss < start_training_generator_loss_threshold
        and loss_delta > switch_training_loss_delta
    ):
        print("Not switching who trains, discriminator still learning")
    elif (
        should_train_generator
        and this_generator_loss > start_training_discriminator_loss_threshold
        and loss_delta > switch_training_loss_delta
    ):
        print("Not switching who trains, generator still learning")
    else:
        should_train_generator = not should_train_generator
        should_train_discriminator = not should_train_discriminator
        print("Switching who trains")
    print(f"{should_train_generator=}, {should_train_discriminator=}")
    return should_train_generator, should_train_discriminator


def train(
    generator: tf.keras.models.Model,
    discriminator: tf.keras.models.Model,
    generator_optimizer: "Optimizer",
    discriminator_optimizer: "Optimizer",
    model_dir: str,
    checkpoint: tf.train.Checkpoint,
    checkpoint_prefix: str,
    dataset_path: str,
    val_path: str,
    train_crop_shape: Tuple[int, int, int],
    num_examples_to_generate: int,
    batch_size: int,
    latent_dim: int,
    epochs: int = 20000,
    buffer_size: int = 1000,
    generator_loss_stop_training_threshold: float = 1.0,
    generator_loss_start_training_threshold: float = 0.1,
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
        discriminator_loss_start_training_threshold: Only switch to training
                                                     discriminator if its loss is higher
                                                     than this
        discriminator_loss_stop_training_threshold: Only switch to training autoencoder
                                                    if discriminator loss is lower than
                                                    this
        continue_from_checkpoint: Restore weights from checkpoint file if given, start
                                  from scratch otherwise.
        debug: Don't die if code not committed (for testing)
    """
    # Skip git checks when debugging
    if not debug:

        # Die if there are uncommitted changes in the repo
        if modified_files_in_git_repo():
            sys.exit(1)

        # Write commit hash to model directory
        os.makedirs(model_dir, exist_ok=True)
        write_commit_hash_to_model_dir(model_dir)

    # Training dataset
    train_images = (
        tf.data.Dataset.from_generator(
            PixelArtSigmoidDataset(
                dataset_path=dataset_path, crop_shape=train_crop_shape
            ),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Make progress dir and reproductions dir for outputs
    consistent_generations_dir = os.path.join(model_dir, "consistent_generations")
    random_generations_dir = os.path.join(model_dir, "random_generations")
    os.makedirs(
        consistent_generations_dir,
        exist_ok=True,
    )
    os.makedirs(
        random_generations_dir,
        exist_ok=True,
    )

    # Set starting and end epoch according to whether we are continuing training
    if continue_from_checkpoint is not None:
        epoch_start = 15 * int(
            continue_from_checkpoint.strip()[continue_from_checkpoint.rfind("-") + 1 :]
        )
    else:
        epoch_start = 0
    epoch_stop = epoch_start + epochs

    # Define metrics
    generator_loss_metric = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
    discriminator_loss_metric = tf.keras.metrics.Mean(
        "discriminator_loss", dtype=tf.float32
    )

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(model_dir, "logs", "gradient_tape", current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Use the same seed throughout training, to see what the model does with the same
    # input as it trains.
    consistent_seed = tf.random.normal(
        shape=[num_examples_to_generate, GENERATOR_LATENT_DIM]
    )

    # Start by training both networks
    should_train_generator = False
    should_train_discriminator = True
    last_generator_loss = 0

    for epoch in range(epoch_start, epoch_stop):
        start = time.time()

        for step, image_batch in enumerate(train_images):
            # Perform training step
            train_step(
                images=image_batch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                loss_metrics={
                    "generator_loss": generator_loss_metric,
                    "discriminator_loss": discriminator_loss_metric,
                },
                should_train_generator=should_train_generator,
                should_train_discriminator=should_train_discriminator,
            )

            if step % 5 == 0:
                print(
                    UPDATE_TEMPLATE.format(
                        # Epoch/step info
                        epoch=epoch + 1,
                        step=step,
                        epoch_time=time.time() - start,
                        # Generator loss metric
                        generator_loss=generator_loss_metric.result(),
                        # Discriminator loss metric
                        discriminator_loss=discriminator_loss_metric.result(),
                    )
                )

        # Produce demo output every epoch the generator trains
        consistent_generated, random_generated = generate_and_save_images(
            generator=generator,
            epoch=epoch + 1,
            consistent_input=consistent_seed,
            consistent_generations_dir=consistent_generations_dir,
            random_generations_dir=random_generations_dir,
        )

        # Write to logs
        with summary_writer.as_default():
            # Learning rates
            tf.summary.scalar(
                "Generator learning rate",
                # autoencoder_optimizer.learning_rate,
                generator_optimizer.learning_rate(epoch * step),
                step=epoch,
            )
            tf.summary.scalar(
                "Discriminator learning rate",
                # discriminator_optimizer.learning_rate,
                discriminator_optimizer.learning_rate(epoch * step),
                step=epoch,
            )

            # Losses
            tf.summary.scalar(
                "Generator loss metric", generator_loss_metric.result(), step=epoch
            )
            tf.summary.scalar(
                "Discriminator loss metric",
                discriminator_loss_metric.result(),
                step=epoch,
            )

            # Example outputs
            tf.summary.image("Consistent generation", consistent_generated, step=epoch)
            tf.summary.image("Random generation", random_generated, step=epoch)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Also close all pyplot figures. It is expensive to do this every epoch
            plt.close("all")

        # Switch who trains if appropriate
        this_generator_loss = generator_loss_metric.result()
        should_train_generator, should_train_discriminator = decide_who_trains(
            should_train_generator=should_train_generator,
            should_train_discriminator=should_train_discriminator,
            this_generator_loss=this_generator_loss,
            last_generator_loss=last_generator_loss,
        )
        last_generator_loss = this_generator_loss

        # Reset metrics every epoch
        generator_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()

        # Try to find memory leak
        print(
            "\n".join(
                f"{key}: {value}" for key, value in sorted_locals(locals()).items()
            ),
            file=sys.stderr,
        )


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


def main(
    mode: str,
    model_dir: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (128, 128, 3),
    buffer_size: int = 1000,
    batch_size: int = 128,
    latent_dim: int = 128,
    num_examples_to_generate: int = 1,
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
    # STEPS_PER_EPOCH = 190  # expanded pixel_art - 128 crops - minibatch size 64 - no aug
    STEPS_PER_EPOCH = 85  # expanded pixel_art - 128 crops - minibatch size 128 - no aug

    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=[STEPS_PER_EPOCH * epoch for epoch in (30, 200)],
    #     values=[1e-4, 7e-5, 3e-5],
    #     name=None,
    # )
    generator_lr = LRS(
        max_lr=3e-4,
        min_lr=3e-5,
        start_decay_epoch=1000,
        stop_decay_epoch=3000,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    discriminator_lr = LRS(
        max_lr=1e-5,
        min_lr=1e-6,
        start_decay_epoch=50,
        stop_decay_epoch=3000,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    # lr = tfa.optimizers.CyclicalLearningRate(
    #     initial_learning_rate=1e-5,
    #     maximal_learning_rate=3e-4,
    #     scale_fn=lambda x: 1 / (1.1 ** (x - 1)),
    #     step_size=3 * STEPS_PER_EPOCH,
    # )

    generator = generator_model()
    discriminator = discriminator_model(input_shape=train_crop_shape)
    # autoencoder_optimizer = tf.keras.optimizers.Adam(lr)
    # discriminator_optimizer = tf.keras.optimizers.Adam(lr)
    generator_optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-7, learning_rate=generator_lr
    )
    discriminator_optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-5, learning_rate=discriminator_lr
    )
    # optimizer = tf.keras.optimizers.Adam(clr)
    # step = tf.Variable(0, trainable=False)
    # optimizer = tfa.optimizers.AdamW(
    #     weight_decay=clr, learning_rate=clr
    # )

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )

    # Restore model from checkpoint
    if continue_from_checkpoint is not None:
        checkpoint.restore(continue_from_checkpoint)

    if mode == "train":
        train(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
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
            latent_dim=latent_dim,
            num_examples_to_generate=num_examples_to_generate,
            continue_from_checkpoint=continue_from_checkpoint,
            debug=debug,
        )
    elif mode == "generate":
        generate(
            generator=generator,
            decoder_input=decoder_input,
            flood_shape=flood_generate_shape,
            latent_dim=latent_dim,
            save_output=save_generator_output,
            sample=sample,
        )
    elif mode == "view-latent-space":
        view_latent_space(generator=generator.decoder_)


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
