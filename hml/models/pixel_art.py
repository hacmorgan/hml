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
import tkinter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageTk
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_gan as tfgan


from hml.architectures.convolutional.classifiers import (
    pixel_art_discriminator,
    dcgan_paper_discriminator,
    dcgan_paper_discriminator_first_conv_unstrided,
    dcgan_paper_discriminator_more_layers,
)
from hml.architectures.convolutional.generators import (
    pixel_art_generator,
    dcgan_paper_generator,
    dcgan_paper_generator_more_layers,
)
from hml.data_pipelines.unsupervised.pixel_art import PixelArtDataset


MODES_OF_OPERATION = ("train", "generate", "discriminate", "view-latent-space")

UPDATE_TEMPLATE = """
Epoch: {epoch}
Step: {step}
Time: {epoch_time}
Generator loss: {generator_loss}
Discriminator Loss: {discriminator_loss}
"""


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


global generator_
global generated_image
global latent_input
global input_neuron
global latent_input_canvas
global generator_output_canvas


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor):
    """
    Custom loss function for the discriminator

    Args:
        loss_fn: Function to compute loss on an image
        real_output: The discriminator's output for a real image
        fake_output: The discriminator's output for a fake image
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output: tf.Tensor):
    """
    Custom loss function for the generator

    The generator wins if the discriminator thinks its output is real (i.e. all ones).
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def gradient_penalty_wasserstein(
    real_data: tf.Tensor,
    fake_data: tf.Tensor,
    discriminator: tf.keras.Sequential,
):
    """
    Gradent penalty term for wasserstein discriminator (critic)
    """
    batch_size = real_data.shape[0]
    if fake_data.shape[0] != batch_size:
        fake_data = fake_data[batch_size, ...]

    # Epsilon must be broadcastable to shape of real_data and fake_data
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_data = real_data + epsilon * (fake_data - real_data)

    # Compute gradient
    d_interpolated = discriminator(interpolated_data)

    # L2 norm is calculated on each sample
    grad_d_interpolated = tf.gradients(d_interpolated, [interpolated_data])[0]
    slopes = tf.sqrt(
        1e-8 + tf.reduce_sum(tf.square(grad_d_interpolated), axis=[1, 2, 3])
    )
    return tf.reduce_mean((slopes - 1.0) ** 2)


def discriminator_loss_wasserstein(
    real_output: tf.Tensor,
    fake_output: tf.Tensor,
    real_image: tf.Tensor,
    fake_image: tf.Tensor,
    discriminator: tf.keras.Sequential,
    regularisation_lambda: float = 1e1,
):
    """
    Wasserstein loss function for the discriminator

    The wasserstein loss function is intended to prevent vanishing gradients. Using a
    wasserstein loss (making this a wasserstein GAN, or WGAN) means the discriminator's
    output cannot be used to detect real or fake by a 0.5 threshold, so we call it a
    critic instead.

    Args:
        loss_fn: Function to compute loss on an image
        real_output: The discriminator's output for a real image
        fake_output: The discriminator's output for a fake image
    """
    if real_output.shape[0] < fake_output.shape[0]:
        fake_output = fake_output[real_output.shape[0], :]
    return (
        tf.reduce_mean(fake_output)
        - tf.reduce_mean(real_output)
        # + regularisation_lambda
        # * gradient_penalty_wasserstein(real_image, fake_image, discriminator)
    )


def generator_loss_wasserstein(fake_output: tf.Tensor):
    """
    Wasserstein loss function for the generator

    The wasserstein loss function is intended to prevent vanishing gradients.

    The generator wins if the discriminator thinks its output is real (i.e. all ones).
    """
    return -tf.reduce_mean(fake_output)
    # return tfgan.losses.wasserstein_generator_loss(fake_output)


def generate_and_save_images(model, epoch, test_input, model_dir: str):
    """
    Generate and save images
    """
    predictions = model(test_input, training=False)

    # Clear the figure
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
    generator: tf.keras.Sequential,
    generator_optimizer: "Optimizer",
    generator_loss_metric: tf.keras.metrics,
    should_train_generator: bool,
    discriminator: tf.keras.Sequential,
    discriminator_optimizer: "Optimizer",
    discriminator_loss_metric: tf.keras.metrics,
    should_train_discriminator: bool,
    batch_size: int,
    noise_size: int,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        generator: Generator model
        generator_optimizer: Optimizer for generator model
        generator_loss: Metric for logging generator loss
        should_train_generator: Update generator weights if True, do nothing otherwise
        discriminator: Discriminator model
        discriminator_optimizer: Optimizer for discriminator model
        discriminator_loss: Metric for logging discriminator loss
        should_train_discriminator: Update discriminator weights if True, do nothing
                                    otherwise
        batch_size: Number of training examples in a batch
        noise_size: Length of input noise vector
    """
    noise = tf.random.normal([batch_size, noise_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        # gen_loss = generator_loss_wasserstein(fake_output)
        # disc_loss = discriminator_loss_wasserstein(
            # real_output, fake_output, images, generated_images, discriminator
        # )

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    if should_train_generator:
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
    if should_train_discriminator:
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

    generator_loss_metric(gen_loss)
    discriminator_loss_metric(disc_loss)


def train(
    generator: tf.keras.Sequential,
    discriminator: tf.keras.Sequential,
    generator_optimizer: "Optimizer",
    discriminator_optimizer: "Optimizer",
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
    generator_loss_metric = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
    discriminator_loss_metric = tf.keras.metrics.Mean(
        "discriminator_loss", dtype=tf.float32
    )

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    generator_log_dir = os.path.join(
        model_dir, "logs", "gradient_tape", current_time, "generator"
    )
    discriminator_log_dir = os.path.join(
        model_dir, "logs", "gradient_tape", current_time, "discriminator"
    )
    generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
    discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)

    # Use the same seed throughout training, to see what the model does with the same input as it trains.
    seed = tf.random.normal([num_examples_to_generate, latent_dim])

    # Start by training both networks
    should_train_generator = True
    should_train_discriminator = True

    # track steps for tensorboard
    total_steps = 0

    for epoch in range(epoch_start, epoch_stop):
        start = time.time()

        for step, image_batch in enumerate(train_images):
            # Perform training step
            train_step(
                images=image_batch,
                generator=generator,
                generator_optimizer=generator_optimizer,
                generator_loss_metric=generator_loss_metric,
                should_train_generator=should_train_generator,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                discriminator_loss_metric=discriminator_loss_metric,
                should_train_discriminator=should_train_discriminator,
                batch_size=batch_size,
                noise_size=latent_dim,
            )

            with generator_summary_writer.as_default(), discriminator_summary_writer.as_default():
                tf.summary.scalar(
                    "generator_loss", generator_loss_metric.result(), step=total_steps
                )
                tf.summary.scalar(
                    "discriminator_loss", discriminator_loss_metric.result(), step=total_steps
                )
                total_steps += 1

            if step % 5 == 0:
                print(
                    UPDATE_TEMPLATE.format(
                        epoch=epoch + 1,
                        step=step,
                        epoch_time=time.time() - start,
                        generator_loss=generator_loss_metric.result(),
                        discriminator_loss=discriminator_loss_metric.result(),
                    )
                )

        # Produce demo output every epoch the generator trains
        if should_train_generator:
            generate_and_save_images(generator, epoch + 1, seed, model_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Write our last epoch down in case we want to continue
            with open(epoch_log_file, "w", encoding="utf-8") as epoch_log:
                epoch_log.write(str(epoch))

        # Alternate who can train periodically
        # After the initial period of training both networks, we alternate who gets to train
        if epoch % epochs_per_turn == 0:
            if should_train_discriminator == should_train_generator:
                should_train_generator = False
                should_train_discriminator = True
            elif (should_train_discriminator and discriminator_loss_metric.result() > generator_loss_metric.result()) or (should_train_generator and generator_loss_metric.result() > discriminator_loss_metric.result()):
                print("Not switching who trains, insufficient progress made")
            else:
                should_train_generator = not should_train_generator
                should_train_discriminator = not should_train_discriminator
            print(
                f"Switching who trains: {should_train_generator=}, {should_train_discriminator=}"
            )

        # Reset metrics every epoch
        generator_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()


def generate(
    generator: tf.keras.Sequential,
    generator_input: Optional[str] = None,
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
    if generator_input is not None:
        input_raw = tf.io.read_file(generator_input)
        input_decoded = tf.image.decode_image(input_raw)
        latent_input = tf.reshape(input_decoded, [1, 100])
    else:
        latent_input = tf.random.normal([1, 100])
    i = 0
    while True:
        generated_rgb_image = np.array(
            (generator(latent_input, training=False)[0, :, :, :] * 127.5 + 127.5)
        ).astype(np.uint8)
        # generated_rgb_image = cv2.cvtColor(generated_hsv_image, cv2.COLOR_HSV2RGB)
        plt.cla()
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
    INIT_LR = 1e-5
    MAX_LR = 2e-4
    STEPS_PER_EPOCH = 27520 / batch_size

    # generator = pixel_art_generator.model()
    # generator = dcgan_paper_generator.model()
    generator = dcgan_paper_generator_more_layers.model()
    generator_optimizer = tf.keras.optimizers.Adam(start_lr, beta_1=0.5)
    # generator_optimizer = tfa.optimizers.AdamW(
    #     weight_decay=3e-7, learning_rate=start_lr, beta_1=0.5
    # )
    # generator_optimizer = tf.keras.optimizers.SGD(clr)

    # discriminator = pixel_art_discriminator.model()
    # discriminator = dcgan_paper_discriminator.model()
    # discriminator = dcgan_paper_discriminator_first_conv_unstrided.model()
    discriminator = dcgan_paper_discriminator_more_layers.model()
    discriminator_optimizer = tf.keras.optimizers.Adam(start_lr)
    # discriminator_optimizer = tfa.optimizers.AdamW(
    #     # weight_decay=3e-7,
    #     weight_decay=1e-4,
    #     learning_rate=start_lr
    # )
    # # clr = tfa.optimizers.CyclicalLearningRate(
    #     initial_learning_rate=INIT_LR,
    #     maximal_learning_rate=MAX_LR,
    #     scale_fn=lambda x: x * (1-1e-3),  # very slowly decaying triangle
    #     # scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    #     step_size=7 * STEPS_PER_EPOCH,  # odd number to switch who gets the highest LR
    # )
    # discriminator_optimizer = tf.keras.optimizers.SGD(clr)
    # piecewise_decay_lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=[100 * STEPS_PER_EPOCH, 500 * STEPS_PER_EPOCH],
    #     values=[2e-4, 1e-4, 5e-5],
    # )
    # discriminator_optimizer = tf.keras.optimizers.SGD(piecewise_decay_lr)

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
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
