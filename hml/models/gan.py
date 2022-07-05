#!/usr/bin/env python3


"""
Train a GAN
"""


__author__ = "Hamish Morgan"


from typing import Dict, Optional, Tuple

import datetime
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
import tensorflow_addons as tfa

# from hml.architectures.convolutional.generators.fhd_generator import (
#     generator,
#     Generator,
# )
# from hml.architectures.convolutional.discriminators.fhd_discriminator import (
#     discriminator,
#     Discriminator,
# )
from hml.architectures.convolutional.generators.generator import generator
from hml.architectures.convolutional.discriminators.discriminator import (
    discriminator,
    LATENT_SHAPE_WIDE,
)

from hml.data_pipelines.unsupervised.upscale_collage import UpscaleDataset

from hml.models import vae

from hml.util.loss import compute_generator_loss, compute_discriminator_loss
from hml.util.gan import decide_who_trains
from hml.util.git import modified_files_in_git_repo, write_commit_hash_to_model_dir


UPDATE_TEMPLATE = """
Epoch: {epoch}    Step: {step}    Time: {epoch_time:.2f}

Generator loss:     {generator_loss:>6.4f}
Discriminator loss: {discriminator_loss:>6.4f}
Currently training: {train_target}
"""


# class Model(tf.keras.models.Model):
class Model:
    """
    Generative Adversarial Network
    """

    def __init__(
        self,
        latent_dim: int = 256,
        # input_shape: Tuple[int, int, int] = (72, 128, 3),
        # input_shape: Tuple[int, int, int] = (1152, 2048, 3),  # stride 2
        input_shape: Tuple[int, int, int] = (2187, 3888, 3),  # stride 3
        conv_filters: int = 128,
        checkpoint: Optional[str] = None,
        vae_checkpoint: Optional[str] = None,
        save_frequency: int = 50,
        # save_frequency: int = 1,
    ) -> "Model":
        """
        Construct the GAN

        Args:
            latent_dim: Dimension of latent distribution (i.e. size of encoder input)
        """
        # Initialise the base class
        super().__init__()

        # High level parameters
        self.latent_dim_ = latent_dim
        self.input_shape_ = input_shape
        self.conv_filters_ = conv_filters
        self.steps_per_epoch_ = 50
        # self.steps_per_epoch_ = 2
        self.checkpoint_path_ = checkpoint
        self.vae_checkpoint_path_ = vae_checkpoint
        self.save_frequency_ = save_frequency

        # Networks
        # self.generator_ = Generator(
        #     latent_dim=latent_dim, conv_filters=self.conv_filters_
        # )
        self.generator_ = generator(
            output_shape=self.input_shape_,
            latent_dim=self.latent_dim_,
            conv_filters=self.conv_filters_,
            latent_shape=LATENT_SHAPE_WIDE,
            strides=3,
        )
        # self.discriminator_ = Discriminator(
        #     input_shape=input_shape, conv_filters=self.conv_filters_
        # )
        self.discriminator_ = discriminator(
            input_shape=self.input_shape_,
            conv_filters=self.conv_filters_,
            latent_shape=LATENT_SHAPE_WIDE,
            strides=3,
        )

        # Learning rates
        self.generator_lr_ = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-6,
            decay_steps=self.steps_per_epoch_ * 20000,
            decay_rate=0.9,
        )
        self.discriminator_lr_ = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=self.steps_per_epoch_ * 20000,
            decay_rate=0.9,
        )

        # Optimizers
        self.generator_optimizer_ = tf.keras.optimizers.Adam(
            learning_rate=self.generator_lr_, beta_1=0.5
        )
        self.discriminator_optimizer_ = tf.keras.optimizers.Adam(
            learning_rate=self.discriminator_lr_, beta_1=0.5
        )
        # self.generator_.custom_compile(optimizer=self.generator_optimizer_)
        # self.discriminator_.custom_compile(optimizer=self.discriminator_optimizer_)

        # Checkpoints
        self.checkpoint_ = tf.train.Checkpoint(
            discriminator=self.discriminator_,
            discriminator_optimizer=self.discriminator_optimizer_,
            generator=self.generator_,
            generator_optimizer=self.generator_optimizer_,
        )

        # Load generator weights from VAE decoder
        if self.vae_checkpoint_path_ is not None:
            self.load_vae_weights(checkpoint_path=self.vae_checkpoint_path_)

        # Restore model from checkpoint
        if self.checkpoint_path_ is not None:
            self.checkpoint_.restore(self.checkpoint_path_)

    def load_vae_weights(self, checkpoint_path: str) -> None:
        """
        Load a vae model from a checkpoint to use its decoder's weights as pretrained
        weights for the generator

        Args:
            checkpoint_path: Path of AE checkpoint
        """
        vae_model = vae.Model(
            latent_dim=self.latent_dim_,
            input_shape=self.input_shape_,
            conv_filters=self.conv_filters_,
            checkpoint=checkpoint_path,
        )
        self.generator_.set_weights(vae_model.net.decoder_.get_weights())
        del vae_model

    def custom_compile(
        self,
    ) -> None:
        """
        Compile the model
        """
        super().compile()
        self.generator_.custom_compile(optimizer=self.generator_optimizer_)
        self.discriminator_.custom_compile(optimizer=self.discriminator_optimizer_)

    # @tf.function
    def _train_step(
        self,
        images: tf.Tensor,
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
        # Generate a new seed to generate a set of new images
        noise = tf.random.normal([tf.shape(images)[0], self.latent_dim_])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:

            # Generate a new full-size image
            generated_images = self.generator_(noise, training=True)

            # Pass real and fake images through discriminator
            real_output = self.discriminator_(images, training=True)
            generated_output = self.discriminator_(generated_images, training=True)

            # Compute losses
            (
                discriminator_loss,
                real_loss,
                generated_loss,
            ) = compute_discriminator_loss(real_output, generated_output)
            generator_loss = compute_generator_loss(generated_output)

        discriminator_gradients = disc_tape.gradient(
            discriminator_loss, self.discriminator_.trainable_variables
        )
        generator_gradients = gen_tape.gradient(
            generator_loss, self.generator_.trainable_variables
        )

        # Compute gradients from losses and update weights
        if self.should_train_discriminator_:
            self.discriminator_optimizer_.apply_gradients(
                zip(discriminator_gradients, self.discriminator_.trainable_variables)
            )
        if self.should_train_generator_:
            self.generator_optimizer_.apply_gradients(
                zip(generator_gradients, self.generator_.trainable_variables)
            )

        # Log losses to their respective metrics
        loss_metrics["discriminator_loss"](discriminator_loss)
        loss_metrics["discriminator_loss_real"](real_loss)
        loss_metrics["discriminator_loss_generated"](generated_loss)
        loss_metrics["generator_loss"](generator_loss)

    def generate_and_save_images(
        self,
        epoch: int,
        test_input: tf.Tensor,
        progress_dir: str,
    ) -> tf.Tensor:
        """
        Generate and save images
        """
        predictions_raw = self.generator_(test_input)
        predictions = (
            (predictions_raw.numpy() * 255).astype(np.uint8).reshape(self.input_shape_)
        )
        output_path = os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png")
        PIL.Image.fromarray(predictions).save(output_path)
        return predictions_raw

    def _train(
        self,
        model_dir: str,
        train_path: str,
        val_path: str,
        epochs: int = 20000,
        batch_size: int = 1,
        num_examples_to_generate: int = 1,
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
            save_frequency: How many epochs between model saves
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

        # Pixel Art dataset
        train_images = (
            tf.data.Dataset.from_generator(
                UpscaleDataset(
                    dataset_path=train_path,
                    output_shape=self.input_shape_,
                    num_examples=self.steps_per_epoch_,
                ),
                output_signature=tf.TensorSpec(
                    shape=self.input_shape_, dtype=tf.float32
                ),
            )
            .batch(batch_size)
            .cache()
            .shuffle(10)
            # .shuffle(self.steps_per_epoch_)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Make progress dir and reproductions dir for outputs
        os.makedirs(
            generated_same_dir := os.path.join(model_dir, "generated_same"),
            exist_ok=True,
        )
        os.makedirs(
            generated_new_dir := os.path.join(model_dir, "generated_new"), exist_ok=True
        )

        # Set up checkpoints
        checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        # Set start and end epoch according to whether we are continuing training
        if self.checkpoint_path_ is not None:
            epoch_start = self.save_frequency_ * int(
                self.checkpoint_path_.strip()[self.checkpoint_path_.rfind("-") + 1 :]
            )
        else:
            epoch_start = 0
        epoch_stop = epoch_start + epochs

        # Define metrics
        generator_loss_metric = tf.keras.metrics.Mean(
            "generator_loss", dtype=tf.float32
        )
        discriminator_loss_metric = tf.keras.metrics.Mean(
            "discriminator_loss", dtype=tf.float32
        )
        discriminator_loss_real_metric = tf.keras.metrics.Mean(
            "discriminator_loss_real", dtype=tf.float32
        )
        discriminator_loss_generated_metric = tf.keras.metrics.Mean(
            "discriminator_loss_generated", dtype=tf.float32
        )

        # Set up logs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(model_dir, "logs", "gradient_tape", current_time)
        summary_writer = tf.summary.create_file_writer(log_dir)

        # Use the same seed throughout training, to see what the model does with the
        # same input as it trains.
        seed = tf.random.normal(shape=[num_examples_to_generate, self.latent_dim_])

        # Start by training both networks
        self.should_train_generator_ = False
        self.should_train_discriminator_ = True
        last_generator_loss = 0

        for epoch in range(epoch_start, epoch_stop):
            start = time.time()

            if self.should_train_discriminator_ and self.should_train_generator_:
                train_target = "both"
            elif self.should_train_discriminator_:
                train_target = "discriminator"
            else:
                train_target = "generator"

            for step, image_batch in enumerate(train_images):
                # Perform training step
                self._train_step(
                    images=image_batch,
                    loss_metrics={
                        "generator_loss": generator_loss_metric,
                        "discriminator_loss": discriminator_loss_metric,
                        "discriminator_loss_real": discriminator_loss_real_metric,
                        "discriminator_loss_generated": discriminator_loss_generated_metric,
                    },
                )

                # print(f"{tf.reduce_mean(self.generator_.trainable_weights)}")
                # print(f"{tf.reduce_mean(self.discriminator_.trainable_weights)}")

                if step % 5 == 0:
                    print(
                        UPDATE_TEMPLATE.format(
                            # Epoch/step info
                            epoch=epoch + 1,
                            step=step,
                            epoch_time=time.time() - start,
                            # Autoencoder loss metrics
                            generator_loss=generator_loss_metric.result(),
                            discriminator_loss=discriminator_loss_metric.result(),
                            train_target=train_target,
                        )
                    )

            # Produce demo output from the same seed and form a new seed each time
            generated_same = self.generate_and_save_images(
                epoch + 1, seed, generated_same_dir
            )
            generated_new = self.generate_and_save_images(
                epoch + 1,
                tf.random.normal(shape=[num_examples_to_generate, self.latent_dim_]),
                generated_new_dir,
            )

            # Write to logs
            with summary_writer.as_default():

                # Learning rates
                tf.summary.scalar(
                    "Generator learning rate",
                    # autoencoder_optimizer.learning_rate,
                    self.generator_optimizer_.learning_rate(epoch * step),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Discriminator learning rate",
                    # discriminator_optimizer.learning_rate,
                    self.discriminator_optimizer_.learning_rate(epoch * step),
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
                tf.summary.scalar(
                    "Discriminator loss real metric",
                    discriminator_loss_real_metric.result(),
                    step=epoch,
                )
                tf.summary.scalar(
                    "Discriminator loss generated metric",
                    discriminator_loss_generated_metric.result(),
                    step=epoch,
                )

                # Example outputs
                tf.summary.image("Consistent generation", generated_same, step=epoch)
                tf.summary.image("Random generation", generated_new, step=epoch)
                tf.summary.image("Training example", image_batch[0:1], step=epoch)

            # Save the model every 2 epochs
            if (epoch + 1) % self.save_frequency_ == 0:
                self.checkpoint_.save(file_prefix=checkpoint_prefix)

                # Also close all pyplot figures. It is expensive to do this every epoch
                plt.close("all")

            # Switch who trains if appropriate
            this_generator_loss = generator_loss_metric.result()
            (
                self.should_train_generator_,
                self.should_train_discriminator_,
            ) = decide_who_trains(
                should_train_generator=self.should_train_generator_,
                should_train_discriminator=self.should_train_discriminator_,
                this_generator_loss=this_generator_loss,
                last_generator_loss=last_generator_loss,
                switch_training_loss_delta=0.01,
                # start_training_discriminator_loss_threshold=1.0,
                # start_training_generator_loss_threshold=2.0,
            )
            last_generator_loss = this_generator_loss
            # if epoch % 4 == 0:
            #     self.should_train_discriminator_ = not self.should_train_discriminator_
            #     self.should_train_generator_ = not self.should_train_generator_
            # else:
            #     self.should_train_discriminator_ = False
            #     self.should_train_generator_ = True

            # Reset metrics every epoch
            discriminator_loss_metric.reset_states()
            discriminator_loss_real_metric.reset_states()
            discriminator_loss_generated_metric.reset_states()
            generator_loss_metric.reset_states()

    def generate(
        autoencoder: tf.keras.models.Model,
        decoder_input: Optional[str] = None,
        latent_dim: int = 50,
        save_output: bool = False,
        num_generations: Optional[int] = None,
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
            latent_input = tf.random.normal([1, latent_dim])
        i = 0
        while True:
            output = autoencoder.net.decoder_(latent_input)
            generated_rgb_image = np.array((output[0, :, :, :] * 255.0)).astype(
                np.uint8
            )
            # generated_rgb_image = cv2.cvtColor(generated_hsv_image, cv2.COLOR_HSV2RGB)
            if save_output:
                save_name = f"generated_{i}.png"
                cv2.imwrite(
                    save_name, cv2.cvtColor(generated_rgb_image, cv2.COLOR_BGR2RGB)
                )
                print(f"image generated to: generated_{i}.png")
                i += 1
            else:
                plt.close("all")
                plt.imshow(generated_rgb_image)
                plt.axis("off")
                plt.show()
            if num_generations is not None and i >= num_generations:
                break
            input("press enter to generate another image")
            latent_input = tf.random.normal([1, latent_dim])
