#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterable, List, Optional, Tuple

import argparse
import datetime
import os
import subprocess
import sys
import threading
import time

# import tkinter

import matplotlib

# matplotlib.use("GTK3Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# import PIL.ImageTk
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_gan as tfgan


from hml.architectures.convolutional.autoencoders.pixel_art_vae import PixelArtVAE
from hml.data_pipelines.unsupervised.pixel_art_sigmoid import PixelArtSigmoidDataset


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")

UPDATE_TEMPLATE = """
Epoch: {epoch}
Step: {step}
Time: {epoch_time}
Loss: {loss}
"""

# # Still TBD whether config is worth it
# DEFAULT_CONFIG = {
#     "train_ds": "/mnt/storage/ml/data/pixel-art/train",
#     "val_ds": "/mnt/storage/ml/data/pixel-art/val",
#     "model_name": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
# }


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
        generated_rgb_image = np.array((predictions[i, :, :, :] * 255)).astype(np.uint8)
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
    autoencoder: tf.keras.Sequential, epoch: int, test_input: tf.Tensor, progress_dir: str
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
    output_path = os.path.join(
        reproductions_dir, f"reproductions_at_epoch_{epoch:04d}.png"
    )
    # threading.Thread(
    #     target=reproduce_save_image, args=(test_images, reproductions, output_path)
    # ).start()
    reproduce_save_image(test_images, reproductions, output_path)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(
    images,
    autoencoder: tf.keras.models.Model,
    optimizer: "Optimizer",
    loss_metric: tf.keras.metrics,
    latent_dim: int,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        autoencoder: Autoencoder model
        optimizer: Optimizer for model
        loss_metric: Metric for logging generator loss
        kl_loss_metric: Metric for logging generator loss
        reconstruction_loss_metric: Metric for logging generator loss
        batch_size: Number of training examples in a batch
        latent_dim: Size of latent space
    """
    # with tf.GradientTape() as tape:
    #     z_mean, z_log_var, z = autoencoder.encoder_(images)
    #     reconstruction = autoencoder.decoder_(z)
    #     # reconstruction_loss = tf.reduce_mean(
    #     #     tf.reduce_sum(bce(images, reconstruction))
    #     #     # tf.reduce_sum(bce(images, reconstruction), axis=(1, 2))
    #     # )
    #     reconstruction_loss = mse(images, reconstruction)
    #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #     total_loss = reconstruction_loss  # + 1e-4 * kl_loss
    # gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    # loss_metric(total_loss)
    # kl_loss_metric(kl_loss)
    # reconstruction_loss_metric(reconstruction_loss)
    with tf.GradientTape() as tape:
        loss = compute_loss(autoencoder, images)
    loss_metric(loss)
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))



def compute_losses(
    autoencoder: tf.keras.models.Model, train_images: tf.Tensor, val_images: tf.Tensor
) -> Tuple[float, float]:
    """
    Compute loss on training and validation data.

    Args:
        autoencoder: Autoencoder model
        train_images: Batch of training data
        val_images: Batch of validation data
    """
    reproduced_train_images = autoencoder.call(train_images, training=False)
    reproduced_val_images = autoencoder.call(val_images, training=False)
    train_loss = mse(train_images, reproduced_train_images)
    val_loss = mse(val_images, reproduced_val_images)
    return train_loss, val_loss


def train(
    autoencoder: tf.keras.models.Model,
    optimizer: "Optimizer",
    model_dir: str,
    checkpoint: tf.train.Checkpoint,
    checkpoint_prefix: str,
    dataset_path: str,
    val_path: str,
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (64, 64, 3),
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

    # Instantiate train and val datasets
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
    val_images = (
        tf.data.Dataset.from_generator(
            PixelArtSigmoidDataset(dataset_path=val_path, crop_shape=train_crop_shape),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
    )

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

    # Set epochs accoridng
    epoch_log_file = os.path.join(model_dir, "epoch_log")
    if continue_from_checkpoint is not None:
        with open(epoch_log_file, "r", encoding="utf-8") as epoch_log:
            epoch_start = int(epoch_log.read().strip()) + 1
    else:
        epoch_start = 0
    epoch_stop = epoch_start + epochs

    # Define our metrics
    loss_metric = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    kl_loss_metric = tf.keras.metrics.Mean("kl_loss", dtype=tf.float32)
    reconstruction_loss_metric = tf.keras.metrics.Mean("reconstruction_loss", dtype=tf.float32)

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(model_dir, "logs", "gradient_tape", current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Use the same seed throughout training, to see what the model does with the same input as it trains.
    seed = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim]
    )

    for epoch in range(epoch_start, epoch_stop):
        start = time.time()

        for step, image_batch in enumerate(train_images):
            # Perform training step
            train_step(
                images=image_batch,
                autoencoder=autoencoder,
                optimizer=optimizer,
                loss_metric=loss_metric,
                latent_dim=latent_dim,
            )

            if step % 5 == 0:
                print(
                    UPDATE_TEMPLATE.format(
                        epoch=epoch + 1,
                        step=step,
                        epoch_time=time.time() - start,
                        loss=loss_metric.result(),
                    )
                )

        # Produce demo output every epoch the generator trains
        generate_and_save_images(autoencoder, epoch + 1, seed, progress_dir)

        # Show some examples of how the model reconstructs its inputs.
        show_reproduction_quality(
            autoencoder, epoch + 1, train_test_images, reproductions_dir
        )
        show_reproduction_quality(
            autoencoder, epoch + 1, val_test_images, val_reproductions_dir
        )

        # Compute loss on training and validation data
        train_loss, val_loss = compute_losses(
            autoencoder, train_test_image_batch, val_test_image_batch
        )

        # Get sample of encoder output
        # z_mean, z_log_var, encoder_output_train = autoencoder.encoder_(train_test_image_batch)
        # z_mean, z_log_var, encoder_output_val = autoencoder.encoder_(val_test_image_batch)

        with summary_writer.as_default():
            tf.summary.scalar(
                "learning rate",
                # optimizer.learning_rate,
                optimizer.learning_rate(epoch * step),
                step=epoch,
            )
            # tf.summary.histogram(
            #     "encoder output train", encoder_output_train, step=epoch
            # )
            # tf.summary.histogram("encoder output val", encoder_output_val, step=epoch)
            # tf.summary.scalar(
            #     "encoder output train: mean",
            #     np.mean(encoder_output_train.numpy()),
            #     step=epoch,
            # )
            # tf.summary.scalar(
            #     "encoder output train: stddev",
            #     np.std(encoder_output_train.numpy()),
            #     step=epoch,
            # )
            # tf.summary.scalar(
            #     "encoder output val: mean",
            #     np.mean(encoder_output_val.numpy()),
            #     step=epoch,
            # )
            # tf.summary.scalar(
            #     "encoder output val: stddev",
            #     np.std(encoder_output_val.numpy()),
            #     step=epoch,
            # )
            tf.summary.scalar("loss metric", loss_metric.result(), step=epoch)
            # tf.summary.scalar("kl loss metric", kl_loss_metric.result(), step=epoch)
            # tf.summary.scalar("reconstruction loss metric", reconstruction_loss_metric.result(), step=epoch)
            tf.summary.scalar("train loss", train_loss, step=epoch)
            tf.summary.scalar("validation loss", val_loss, step=epoch)
            # tf.summary.image("dd", step=epoch)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Also close all pyplot figures. It is expensive to do this every epoch
            plt.close("all")

            # Write our last epoch down in case we want to continue
            with open(epoch_log_file, "w", encoding="utf-8") as epoch_log:
                epoch_log.write(str(epoch))

        # Reset metrics every epoch
        loss_metric.reset_states()


def generate(
    decoder: tf.keras.Sequential,
    decoder_input: Optional[str] = None,
    latent_dim: int = 50,
    save_output: bool = False,
) -> None:
    """
    Generate some pixel art

    Args:
        generator: Generator model
        generator_input: Path to a 10x10 grayscale image to use as input. Random noise
                         used if not given.
        save_output: Save images instead of displaying
    """
    if decoder_input is not None:
        input_raw = tf.io.read_file(decoder_input)
        input_decoded = tf.image.decode_image(input_raw)
        latent_input = tf.reshape(input_decoded, [1, latent_dim])
    else:
        latent_input = tf.random.normal([1, latent_dim])
    i = 0
    while True:
        generated_rgb_image = np.array(
            (decoder(latent_input, training=False)[0, :, :, :] * 255.0)
        ).astype(np.uint8)
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
        latent_input = tf.random.normal([1, latent_dim])


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
        generator_(latent_input, training=False)[0, :, :, :] * 127.5 + 127.5
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
    train_crop_shape: Tuple[int, int, int] = (64, 64, 3),
    buffer_size: int = 20000,
    batch_size: int = 64,
    epochs_per_turn: int = 1,
    latent_dim: int = 200,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
    decoder_input: Optional[str] = None,
    save_generator_output: bool = False,
    debug: bool = False,
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
    # STEPS_PER_EPOCH = 1410  # with x4 augmentation
    STEPS_PER_EPOCH = 705  # with x2 augmentation
    # STEPS_PER_EPOCH = 350  # with no augmentation

    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=1e-4,
        maximal_learning_rate=1e-3,
        scale_fn=lambda x: 1 / (1.2 ** (x - 1)),
        # scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=3 * STEPS_PER_EPOCH,
    )

    autoencoder = PixelArtVAE(latent_dim=latent_dim)
    # optimizer = tf.keras.optimizers.Adam(1e-4)
    # optimizer = tf.keras.optimizers.Adam(clr)
    # step = tf.Variable(0, trainable=False)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=clr, learning_rate=clr
    )

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        autoencoder=autoencoder,
        optimizer=optimizer,
    )

    # Restore model from checkpoint
    if continue_from_checkpoint is not None:
        checkpoint.restore(continue_from_checkpoint)

    if mode == "train":
        train(
            autoencoder=autoencoder,
            optimizer=optimizer,
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
            decoder=autoencoder.decoder_,
            decoder_input=decoder_input,
            latent_dim=latent_dim,
            save_output=save_generator_output,
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
