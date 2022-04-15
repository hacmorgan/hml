#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterable, List, Optional, Tuple

import argparse
import datetime
import os
import sys
import time
# import tkinter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageTk
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_gan as tfgan


from hml.architectures.convolutional.autoencoders.pixel_art_ae import PixelArtAE
from hml.data_pipelines.unsupervised.pixel_art import PixelArtDataset


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")

UPDATE_TEMPLATE = """
Epoch: {epoch}
Step: {step}
Time: {epoch_time}
Loss: {loss}
"""


# This method returns a helper function to compute mean squared error loss
mse = tf.keras.losses.MeanSquaredError()


global generator_
global generated_image
global latent_input
global input_neuron
global latent_input_canvas
global generator_output_canvas


def generate_and_save_images(
    decoder: tf.keras.Sequential, epoch: int, test_input: tf.Tensor, model_dir: str
) -> None:
    """
    Generate and save images
    """
    predictions = decoder(test_input, training=False)

    # Clear the figure (prevents memory leaks)
    plt.cla()

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        generated_rgb_image = np.array(
            (predictions[i, :, :, :] * 127.5 + 127.5)
        ).astype(np.uint8)
        # generated_rgb_image = cv2.cvtColor(generated_hsv_image, cv2.COLOR_HSV2RGB)
        plt.imshow(generated_rgb_image)
        plt.axis("off")

    os.makedirs(progress_dir := os.path.join(model_dir, "progress"), exist_ok=True)
    plt.savefig(os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png"), dpi=250)


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
        batch_size: Number of training examples in a batch
        latent_dim: Size of latent space
    """
    with tf.GradientTape() as tape:
        reproduced_images = autoencoder.call(images)
        loss = mse(images, reproduced_images)
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, autoencoder.trainable_variables)
    )
    loss_metric(loss)  # save loss for plotting


def train(
    autoencoder: tf.keras.models.Model,
    optimizer: "Optimizer",
    model_dir: str,
    checkpoint: tf.train.Checkpoint,
    checkpoint_prefix: str,
    dataset_path: str,
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (64, 64, 3),
    buffer_size: int = 1000,
    batch_size: int = 128,
    epochs_per_turn: int = 1,
    latent_dim: int = 100,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
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
    train_images = (
        tf.data.Dataset.from_generator(
            PixelArtDataset(dataset_path=dataset_path, crop_shape=train_crop_shape),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
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

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        model_dir, "logs", "gradient_tape", current_time
    )
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Use the same seed throughout training, to see what the model does with the same input as it trains.
    seed = tf.random.normal([num_examples_to_generate, latent_dim])

    # track steps for tensorboard
    total_steps = 0

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

            with summary_writer.as_default():
                tf.summary.scalar(
                    "loss", loss_metric.result(), step=total_steps
                )
                total_steps += 1

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
        generate_and_save_images(autoencoder.decoder_, epoch + 1, seed, model_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Write our last epoch down in case we want to continue
            with open(epoch_log_file, "w", encoding="utf-8") as epoch_log:
                epoch_log.write(str(epoch))

        # Reset metrics every epoch
        loss_metric.reset_states()


def generate(
    decoder: tf.keras.Sequential,
    decoder_input: Optional[str] = None,
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
        latent_input = tf.reshape(input_decoded, [1, 100])
    else:
        latent_input = tf.random.normal([1, 100])
    i = 0
    while True:
        generated_rgb_image = np.array(
            (decoder(latent_input, training=False)[0, :, :, :] * 127.5 + 127.5)
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
        latent_input = tf.random.normal([1, 100])


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
    epochs: int = 20000,
    train_crop_shape: Tuple[int, int, int] = (64, 64, 3),
    buffer_size: int = 1000,
    batch_size: int = 64,
    epochs_per_turn: int = 1,
    latent_dim: int = 100,
    num_examples_to_generate: int = 16,
    continue_from_checkpoint: Optional[str] = None,
    generator_input: Optional[str] = None,
    save_generator_output: bool = False,
) -> None:
    """
    Main routine.

    Args:
        mode: One of "train" "generate" "discriminate"
        model_dir: Working dir for this experiment
        dataset_path: Path to directory tree containing training data
        epochs: How many full passes through the dataset to make
        train_crop_shape: Desired shape of training crops from full images
        buffer_size: Number of images to randomly sample from at a time
        batch_size: Number of training examples in a batch
        epochs_per_turn: How long to train one model before switching to the other
        latent_dim: Number of noise inputs to generator
        num_examples_to_generate: How many examples to generate on each epoch
        continue_from_checkpoint: Restore weights from checkpoint file if given, start
                                  from scratch otherwise.
        generator_input: Path to a 10x10 grayscale image, to be used as input to the
                         generator. Noise used if None
        save_generator_output: Save generated images instead of displaying
    """
    start_lr = 2e-4

    autoencoder = PixelArtAE(latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(start_lr)

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
            epochs=epochs,
            train_crop_shape=train_crop_shape,
            buffer_size=buffer_size,
            batch_size=batch_size,
            epochs_per_turn=epochs_per_turn,
            latent_dim=latent_dim,
            num_examples_to_generate=num_examples_to_generate,
            continue_from_checkpoint=continue_from_checkpoint,
        )
    elif mode == "generate":
        generate(
            generator=generator,
            generator_input=generator_input,
            save_output=save_generator_output,
        )
    elif mode == "view-latent-space":
        view_latent_space(generator=generator)


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
        default="./training-data",
        help="Path to dataset directory, containing training images",
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
        continue_from_checkpoint=args.checkpoint,
        generator_input=args.generator_input,
        save_generator_output=args.save_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli_main(get_args()))
