#!/usr/bin/env python3


"""
Train a VAE
"""


__author__ = "Hamish Morgan"


from typing import Dict, Optional, Tuple

import argparse
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

from hml.architectures.convolutional.autoencoders.vae import VAE
from hml.data_pipelines.unsupervised.upscale_collage import UpscaleDataset
from hml.util.git import modified_files_in_git_repo, write_commit_hash_to_model_dir
from hml.util.learning_rate import WingRampLRS


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")


UPDATE_TEMPLATE = """
Epoch: {epoch}    Step: {step}    Time: {epoch_time:.2f}

KL loss:                            {kl_loss:>6.4f}
Reconstruction sharpness loss:      {reconstruction_sharpness_loss:>6.4f}
Generation sharpness loss:          {generation_sharpness_loss:>6.4f}
Total autoencoder loss:             {vae_loss:>6.4f}
"""


# This method returns a helper function to compute mean squared error loss
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()


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
    test_images: tf.Tensor, reproductions: tf.Tensor, output_path: str
) -> None:
    """
    Regenerate some images and save to file.

    This is done in a separate thread to avoid waiting for io.
    """
    for i, (test_image, reproduction) in enumerate(zip(test_images, reproductions)):
        plt.subplot(2, 4, i + 1)
        stacked = tf.concat([test_image, reproduction], axis=0)
        rgb_image = np.array((stacked * 255.0)).astype(np.uint8)
        plt.imshow(rgb_image)
        plt.axis("off")
    plt.savefig(output_path, dpi=250)


def generate_and_save_images(
    autoencoder: tf.keras.Sequential,
    epoch: int,
    test_input: tf.Tensor,
    progress_dir: str,
) -> None:
    """
    Generate and save images
    """
    predictions = autoencoder.sample(test_input)
    output_path = os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png")
    # threading.Thread(
    #     target=generate_save_image, args=(predictions, output_path)
    # ).start()
    generate_save_image(predictions, output_path)


def show_reproduction_quality(
    autoencoder: tf.keras.models.Model,
    epoch: int,
    test_images: tf.Tensor,
    reproductions_dir: str,
) -> None:
    """
    Generate and save images
    """
    reproductions = autoencoder.call(test_images, training=False)
    print(f"{np.mean(reproductions)=}, {np.std(reproductions)=}")
    print(f"{np.mean(test_images)=}, {np.std(test_images)=}")
    output_path = os.path.join(
        reproductions_dir, f"reproductions_at_epoch_{epoch:04d}.png"
    )
    # threading.Thread(
    #     target=reproduce_save_image, args=(test_images, reproductions, output_path)
    # ).start()
    reproduce_save_image(test_images, reproductions, output_path)


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
    images: tf.Tensor,
    alpha: float = 1e0,
    delta: float = 1e-3,
    epsilon: float = 0e0,
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
        zeta: Contribution of smooth boundaries loss to total loss

    Returns:
        Total loss
        VAE (KL divergence) component of loss
        Image sharpness loss on generated images
        Image sharpness loss on reconstructed images
        Reconstructed images (used again to compute loss for training discriminator)
        Generated images (used again to compute loss for training discriminator)
    """
    # Reconstruct a real image
    mean, logvar = vae.encode(images)
    z = vae.reparameterize(mean, logvar)
    x_logit = vae.decode(z)
    reconstructed = vae.sample(z)

    # Generate an image from noise input
    z_gen = tf.random.normal(shape=z.shape)
    generated = vae.sample(z_gen)

    # Compute KL divergence loss
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=images)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

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
        + delta * sharpness_loss_generated
        + epsilon * sharpness_loss_reconstructed
    )
    return (
        loss,
        kl_loss,
        sharpness_loss_generated,
        sharpness_loss_reconstructed,
        reconstructed,
        generated,
    )


@tf.function
def train_step(
    images: tf.Tensor,
    autoencoder: tf.keras.models.Model,
    autoencoder_optimizer: "Optimizer",
    loss_metrics: Dict[str, "Metrics"],
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        autoencoder: Autoencoder model
        autoencoder_optimizer: Optimizer for autoencoder model
        loss_metrics: Dictionary of metrics to log
        batch_size: Number of training examples in a batch
        latent_dim: Size of latent space
    """
    # Compute losses
    with tf.GradientTape() as vae_tape:
        (
            vae_loss,
            kl_loss,
            generation_sharpness_loss,
            reconstruction_sharpness_loss,
            reconstructed_images,
            generated_images,
        ) = compute_vae_loss(
            autoencoder,
            images,
        )

    # Compute gradients from losses
    vae_gradients = vae_tape.gradient(vae_loss, autoencoder.trainable_variables)

    # Update weights
    autoencoder_optimizer.apply_gradients(
        zip(vae_gradients, autoencoder.trainable_variables)
    )

    # Log losses to their respective metrics
    loss_metrics["vae_loss_metric"](vae_loss)
    loss_metrics["kl_loss_metric"](kl_loss)
    loss_metrics["generation_sharpness_loss_metric"](generation_sharpness_loss)
    loss_metrics["reconstruction_sharpness_loss_metric"](reconstruction_sharpness_loss)


def compute_mse_losses(
    autoencoder: tf.keras.models.Model,
    train_images: tf.Tensor,
    val_images: tf.Tensor,
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
    train_loss = mse(train_images, reproduced_train_images)
    val_loss = mse(val_images, reproduced_val_images)
    return train_loss, val_loss


def train(
    autoencoder: tf.keras.models.Model,
    autoencoder_optimizer: "Optimizer",
    model_dir: str,
    checkpoint: tf.train.Checkpoint,
    checkpoint_prefix: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    output_shape: Tuple[int, int, int] = (1152, 2048, 3),
    batch_size: int = 128,
    latent_dim: int = 128,
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
        discriminator_loss_start_training_threshold: Only switch to training
            discriminator if its loss is higher than this
        discriminator_loss_stop_training_threshold: Only switch to training autoencoder
            if discriminator loss is lower than this
        continue_from_checkpoint: Restore weights from checkpoint file if given, start
            from scratch otherwise.
        debug: Don't die if code not committed (for testing)
    """
    # Unless we are debugging, enforce that changes are committed
    if not debug:

        # Die if there are uncommitted changes in the repo
        if modified_files_in_git_repo():
            return

        # Write commit hash to model directory
        os.makedirs(model_dir, exist_ok=True)
        write_commit_hash_to_model_dir(model_dir)

    # Shape of 3x3 blocks context region
    rows, cols, channels = output_shape

    # # Pixel Art dataset
    # train_images = (
    #     tf.data.Dataset.from_generator(
    #         UpscaleDataset(
    #             dataset_path=dataset_path, output_shape=output_shape, num_examples=32
    #         ),
    #         output_signature=(tf.TensorSpec(shape=output_shape, dtype=tf.float32),),
    #     )
    #     .batch(batch_size)
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # val_images = tf.data.Dataset.from_generator(
    # UpscaleDataset(
    #     dataset_path=val_path, output_shape=output_shape, num_examples=32
    # ),
    #     output_signature=(tf.TensorSpec(shape=output_shape, dtype=tf.float32),),
    # ).batch(batch_size)

    train_ds = UpscaleDataset(
        dataset_path=dataset_path, output_shape=output_shape, num_examples=32
    )
    val_ds = UpscaleDataset(
        dataset_path=val_path, output_shape=output_shape, num_examples=32
    )

    train_images = train_ds()
    val_images = val_ds()

    # Save a few images for visualisation
    train_test_image_batch = next(iter(train_images))
    train_test_images = train_test_image_batch[:8, ...]
    val_test_image_batch = next(iter(val_images))
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

    # Set starting and end epoch according to whether we are continuing training
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

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(model_dir, "logs", "gradient_tape", current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Use the same seed throughout training, to see what the model does with the same
    # input as it trains.
    seed = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    for epoch in range(epoch_start, epoch_stop):
        start = time.time()

        # for step, (image_batch, label_batch) in enumerate(train_images):
        for step, image_batch in enumerate(train_ds()):
            # Perform training step
            train_step(
                images=image_batch,
                autoencoder=autoencoder,
                autoencoder_optimizer=autoencoder_optimizer,
                loss_metrics={
                    "vae_loss_metric": vae_loss_metric,
                    "kl_loss_metric": kl_loss_metric,
                    "generation_sharpness_loss_metric": generation_sharpness_loss_metric,
                    "reconstruction_sharpness_loss_metric": reconstruction_sharpness_loss_metric,
                },
            )

            if step % 5 == 0:
                print(
                    UPDATE_TEMPLATE.format(
                        # Epoch/step info
                        epoch=epoch + 1,
                        step=step,
                        epoch_time=time.time() - start,
                        # Autoencoder loss metrics
                        vae_loss=vae_loss_metric.result(),
                        kl_loss=kl_loss_metric.result(),
                        generation_sharpness_loss=generation_sharpness_loss_metric.result(),
                        reconstruction_sharpness_loss=reconstruction_sharpness_loss_metric.result(),
                    )
                )

        # Produce demo output every epoch the generator trains
        generated = generate_and_save_images(autoencoder, epoch + 1, seed, progress_dir)

        # Show some examples of how the model reconstructs its inputs.
        train_reconstructed = show_reproduction_quality(
            autoencoder,
            epoch + 1,
            train_test_images,
            reproductions_dir,
        )
        val_reconstructed = show_reproduction_quality(
            autoencoder,
            epoch + 1,
            val_test_images,
            val_reproductions_dir,
        )

        # Compute reconstruction loss on training and validation data
        train_loss, val_loss = compute_mse_losses(
            autoencoder,
            train_test_image_batch,
            val_test_image_batch,
        )

        # Get sample of encoder output
        z_mean, z_log_var = autoencoder.encode(train_test_image_batch)
        encoder_output_train = autoencoder.reparameterize(z_mean, z_log_var)
        z_mean, z_log_var = autoencoder.encode(val_test_image_batch)
        encoder_output_val = autoencoder.reparameterize(z_mean, z_log_var)

        # Write to logs
        with summary_writer.as_default():

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

            # Learning rates
            tf.summary.scalar(
                "VAE learning rate",
                # autoencoder_optimizer.learning_rate,
                autoencoder_optimizer.learning_rate(epoch * step),
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

            # MSE reconstruction losses
            tf.summary.scalar("MSE train loss", train_loss, step=epoch)
            tf.summary.scalar("MSE validation loss", val_loss, step=epoch)

            # Example outputs
            tf.summary.image(
                "reconstructed train images", train_reconstructed, step=epoch
            )
            tf.summary.image("reconstructed val images", val_reconstructed, step=epoch)
            tf.summary.image("generated images", generated, step=epoch)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Also close all pyplot figures. It is expensive to do this every epoch
            plt.close("all")

        # Reset metrics every epoch
        vae_loss_metric.reset_states()
        kl_loss_metric.reset_states()
        generation_sharpness_loss_metric.reset_states()
        reconstruction_sharpness_loss_metric.reset_states()


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


def main(
    mode: str,
    model_dir: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    output_shape: Tuple[int, int, int] = (1152, 2048, 3),
    buffer_size: int = 1000,
    batch_size: int = 1,
    latent_dim: int = 256,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
    decoder_input: Optional[str] = None,
    save_generator_output: bool = False,
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
    STEPS_PER_EPOCH = 128  # Randomly generated

    autoencoder_lr = WingRampLRS(
        max_lr=1e-4,
        min_lr=1e-5,
        start_decay_epoch=50,
        stop_decay_epoch=1500,
        steps_per_epoch=STEPS_PER_EPOCH,
    )

    autoencoder = VAE(latent_dim=latent_dim, input_shape=output_shape)
    autoencoder_optimizer = tfa.optimizers.AdamW(
        weight_decay=1e-7, learning_rate=autoencoder_lr
    )

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        autoencoder=autoencoder,
        autoencoder_optimizer=autoencoder_optimizer,
    )

    # Restore model from checkpoint
    if continue_from_checkpoint is not None:
        checkpoint.restore(continue_from_checkpoint)

    if mode == "train":
        train(
            autoencoder=autoencoder,
            autoencoder_optimizer=autoencoder_optimizer,
            model_dir=model_dir,
            checkpoint=checkpoint,
            checkpoint_prefix=checkpoint_prefix,
            dataset_path=dataset_path,
            val_path=val_path,
            epochs=epochs,
            output_shape=output_shape,
            batch_size=batch_size,
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
        val_path=args.validation_dataset,
        continue_from_checkpoint=args.checkpoint,
        decoder_input=args.generator_input,
        save_generator_output=args.save_output,
        sample=not args.no_sample,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
