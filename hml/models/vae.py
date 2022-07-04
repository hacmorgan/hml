#!/usr/bin/env python3


"""
Train a VAE
"""


__author__ = "Hamish Morgan"


from typing import Dict, Optional, Tuple

import argparse
import datetime
import io
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from hml.architectures.convolutional.autoencoders.vae import VariationalAutoEncoder

from hml.data_pipelines.unsupervised.upscale_collage import UpscaleDataset

from hml.util.git import modified_files_in_git_repo, write_commit_hash_to_model_dir
from hml.util.learning_rate import WingRampLRS
from hml.util.image import variance_of_laplacian


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


class Model(tf.keras.models.Model):
    """
    Autoencoder with architecture based on DCGAN paper

    The adversarial part of this model is handled in hml.models.pixel_art_avae, so this
    is just a plain VAE
    """

    def __init__(
        self,
        latent_dim: int = 256,
        # input_shape: Tuple[int, int, int] = (1152, 2048, 3),
        input_shape: Tuple[int, int, int] = (2187, 3888, 3),  # stride 3
        conv_filters: int = 128,
        checkpoint: Optional[str] = None,
        save_frequency: int = 50,
    ) -> "Model":
        """
        Construct the autoencoder

        Args:
            latent_dim: Dimension of latent distribution (i.e. size of encoder input)
        """
        # Initialise the base class
        super().__init__()

        # High level parameters
        self.latent_dim_ = latent_dim
        self.input_shape_ = input_shape
        self.net = VariationalAutoEncoder(
            latent_dim=self.latent_dim_,
            input_shape=self.input_shape_,
            conv_filters=conv_filters,
            strides=3,
        )
        self.steps_per_epoch_ = 50
        self.checkpoint_path_ = checkpoint
        self.structure = {
            "encoder_": self.net.encoder_,
            "decoder_": self.net.decoder_,
        }

        # # Networks
        # # self.generator_ = Generator(
        # #     latent_dim=latent_dim, conv_filters=self.conv_filters_
        # # )
        # self.generator_ = generator(
        #     output_shape=self.input_shape_,
        #     latent_dim=self.latent_dim_,
        #     conv_filters=self.conv_filters_,
        #     latent_shape=LATENT_SHAPE_WIDE,
        #     strides=3,
        # )
        # # self.discriminator_ = Discriminator(
        # #     input_shape=input_shape, conv_filters=self.conv_filters_
        # # )
        # self.discriminator_ = discriminator(
        #     input_shape=self.input_shape_,
        #     conv_filters=self.conv_filters_,
        #     latent_shape=LATENT_SHAPE_WIDE,
        #     strides=3,
        # )

        # Configure learning rate
        self.lr_ = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=self.steps_per_epoch_ * 1000,
            decay_rate=0.9,
        )

        # Configure optimiser
        self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=self.lr_)

        # # Compile model
        # self.custom_compile(optimizer=self.optimizer_)

        # Set up checkpoints
        self.checkpoint_ = tf.train.Checkpoint(
            autoencoder=self,
            autoencoder_optimizer=self.optimizer_,
        )

        # Restore model from checkpoint
        if self.checkpoint_path_ is not None:
            self.checkpoint_.restore(self.checkpoint_path_)

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tfp.distributions.MultivariateNormalDiag, tf.Tensor]:
        """
        Run the model

        Args:
            inputs: Input images to be encoded
            training: True if we are training, False otherwise

        Returns:
            Posterior distribution (from encoder outputs)
            Reconstructed image
        """
        return self.net(inputs, training)

    def custom_compile(
        self,
        loss_fn: str = "kl_mse",
        alpha: float = 1.0,
    ) -> None:
        """
        Compile the model

        Args:
            loss_fn: Name of loss function
            alpha: Coefficient to balance divergence and likelihood losses
        """
        super().compile()
        self.optimizer = self.optimizer_
        self.loss_fn = loss_fn
        self.alpha_ = alpha

    def compute_vae_loss(
        self,
        images: tf.Tensor,
        beta: float = 1e0,
        delta: float = 0e1,
        epsilon: float = 0e0,
    ) -> Tuple[float, float, float, float]:
        """
        Compute loss for training VAE

        Args:
            vae: VAE model - should have encode(), decode(), reparameterize(), and sample()
                methods
            discriminator: The discriminator model
            images: Minibatch of training images
            labels: Minibatch of training labels
            beta: Contribution of KL divergence loss to total loss
            gamma: Contribution of regularization loss to total loss
            delta: Contribution of generated image sharpness loss to total loss
            epsilon: Contribution of reconstructed image sharpness loss to total loss

        Returns:
            Total loss
            VAE (KL divergence) component of loss
            Image sharpness loss on generated images
            Image sharpness loss on reconstructed images
        """
        # Reconstruct real image(s)
        posterior, reconstructions = self(images, training=True)

        # Generate image(s) from noise input
        generated = self.net.decoder_(
            tf.random.normal(shape=(tf.shape(images)[0], self.latent_dim_))
        )

        # Create unit gaussian prior
        prior = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0] * self.latent_dim_, scale_diag=[1.0] * self.latent_dim_
        )

        # Compute KL divergence
        divergence = tf.maximum(tfp.distributions.kl_divergence(posterior, prior), 0)

        # Compute likelihood
        likelihood = -tf.reduce_sum(
            tf.keras.metrics.mean_squared_error(images, reconstructions)
        )

        # Compute ELBO loss
        kl_loss = -tf.reduce_mean(likelihood - self.alpha_ * divergence)

        # Sharpness loss on generated images
        sharpness_loss_generated = -(
            variance_of_laplacian(generated, ksize=3)
            + variance_of_laplacian(generated, ksize=5)
            + variance_of_laplacian(generated, ksize=7)
        )

        # Sharpness loss on reconstructed images
        sharpness_loss_reconstructed = -(
            variance_of_laplacian(reconstructions, ksize=3)
            + variance_of_laplacian(reconstructions, ksize=5)
            + variance_of_laplacian(reconstructions, ksize=7)
        )

        # Compute total loss and return
        # loss = (
        #     beta * kl_loss
        #     + delta * sharpness_loss_generated
        #     + epsilon * sharpness_loss_reconstructed
        # )
        loss = kl_loss
        return (
            loss,
            kl_loss,
            sharpness_loss_generated,
            sharpness_loss_reconstructed,
        )

    @tf.function
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
        # Compute losses
        with tf.GradientTape() as tape:
            reg_loss = tf.reduce_sum(self.losses)
            (
                vae_loss,
                kl_loss,
                generation_sharpness_loss,
                reconstruction_sharpness_loss,
            ) = self.compute_vae_loss(images)
            loss = reg_loss + vae_loss

        # Compute gradients from losses
        gradients = tape.gradient(loss, self.net.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

        # Log losses to their respective metrics
        loss_metrics["vae_loss_metric"](vae_loss)
        loss_metrics["kl_loss_metric"](kl_loss)
        loss_metrics["generation_sharpness_loss_metric"](generation_sharpness_loss)
        loss_metrics["reconstruction_sharpness_loss_metric"](
            reconstruction_sharpness_loss
        )

    def generate_and_save_images(
        self,
        epoch: int,
        test_input: tf.Tensor,
        progress_dir: str,
    ) -> tf.Tensor:
        """
        Generate and save images
        """
        predictions = self.net.decoder_(test_input)
        output_path = os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(predictions.numpy(), cv2.COLOR_BGR2RGB))
        return predictions

    def show_reproduction_quality(
        self,
        epoch: int,
        test_images: tf.Tensor,
        reproductions_dir: str,
    ) -> tf.Tensor:
        """
        Generate and save images
        """
        _, reproductions = self(test_images)
        stacked = tf.concat([test_images, reproductions], axis=1)
        output_path = os.path.join(
            reproductions_dir, f"reproductions_at_epoch_{epoch:04d}.png"
        )
        cv2.imwrite(output_path, cv2.cvtColor(stacked.numpy(), cv2.COLOR_BGR2RGB))
        return stacked

    def compute_mse_losses(
        self,
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
        _, reproduced_train_images = self.call(train_images)
        _, reproduced_val_images = self.call(val_images)
        train_loss = mse(train_images, reproduced_train_images)
        val_loss = mse(val_images, reproduced_val_images)
        return train_loss, val_loss

    def _train(
        self,
        model_dir: str,
        train_path: str,
        val_path: str,
        epochs: int = 100,
        # output_shape: Tuple[int, int, int] = (1152, 2048, 3),
        output_shape: Tuple[int, int, int] = (2187, 3888, 3),  # stride 3
        batch_size: int = 1,
        latent_dim: int = 256,
        num_examples_to_generate: int = 1,
        continue_from_checkpoint: Optional[str] = None,
        save_frequency: int = 10,
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

        # Shape of 3x3 blocks context region
        rows, cols, channels = output_shape

        # Pixel Art dataset
        train_images = (
            tf.data.Dataset.from_generator(
                UpscaleDataset(
                    dataset_path=train_path,
                    output_shape=output_shape,
                    num_examples=self.steps_per_epoch_,
                ),
                output_signature=tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            )
            .batch(batch_size)
            .cache()
            .shuffle(10)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_images = (
            tf.data.Dataset.from_generator(
                UpscaleDataset(
                    dataset_path=val_path, output_shape=output_shape, num_examples=1
                ),
                output_signature=tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            )
            .batch(batch_size)
            .cache()
        )

        # Make progress dir and reproductions dir for outputs
        os.makedirs(
            generated_same_dir := os.path.join(model_dir, "generated_same"),
            exist_ok=True,
        )
        os.makedirs(
            generated_new_dir := os.path.join(model_dir, "generated_new"), exist_ok=True
        )
        os.makedirs(
            reproductions_dir := os.path.join(model_dir, "reproductions"), exist_ok=True
        )
        os.makedirs(
            val_reproductions_dir := os.path.join(model_dir, "val_reproductions"),
            exist_ok=True,
        )

        # Set up checkpoints
        checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        # Set starting and end epoch according to whether we are continuing training
        if continue_from_checkpoint is not None:
            epoch_start = save_frequency * int(
                continue_from_checkpoint.strip()[
                    continue_from_checkpoint.rfind("-") + 1 :
                ]
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

        # Save training data from first step
        first_step = True

        for epoch in range(epoch_start, epoch_stop):
            start = time.time()

            for step, image_batch in enumerate(train_images):

                # Save a few images for visualisation
                if first_step:
                    train_test_image_batch = image_batch
                    train_test_images = train_test_image_batch[:8, ...]
                    val_test_image_batch = next(iter(val_images))
                    val_test_images = val_test_image_batch[:8, ...]
                    first_step = False

                # for step, image_batch in enumerate(train_ds()):
                # Perform training step
                self._train_step(
                    images=image_batch,
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

            # Produce demo output from the same seed and form a new seed each time
            generated_same = self.generate_and_save_images(
                epoch + 1, seed, generated_same_dir
            )
            generated_new = self.generate_and_save_images(
                epoch + 1,
                tf.random.normal(shape=[num_examples_to_generate, latent_dim]),
                generated_new_dir,
            )

            # Show some examples of how the model reconstructs its inputs.
            train_reconstructed = self.show_reproduction_quality(
                epoch + 1,
                train_test_images,
                reproductions_dir,
            )
            val_reconstructed = self.show_reproduction_quality(
                epoch + 1,
                val_test_images,
                val_reproductions_dir,
            )

            # Compute reconstruction loss on training and validation data
            train_loss, val_loss = self.compute_mse_losses(
                train_test_image_batch,
                val_test_image_batch,
            )

            # Write to logs
            with summary_writer.as_default():

                # Losses
                tf.summary.scalar(
                    "VAE loss metric", vae_loss_metric.result(), step=epoch
                )
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
                    self.optimizer_.learning_rate(epoch * step),
                    step=epoch,
                )

                # MSE reconstruction losses
                tf.summary.scalar("MSE train loss", train_loss, step=epoch)
                tf.summary.scalar("MSE validation loss", val_loss, step=epoch)

                # Example outputs
                tf.summary.image(
                    "reconstructed train images", train_reconstructed, step=epoch
                )
                tf.summary.image(
                    "reconstructed val images", val_reconstructed, step=epoch
                )
                tf.summary.image(
                    "generated images (same seed)", generated_same, step=epoch
                )
                tf.summary.image(
                    "generated images (new seed)", generated_new, step=epoch
                )

            # Save the model every 2 epochs
            if (epoch + 1) % save_frequency == 0:
                self.checkpoint_.save(file_prefix=checkpoint_prefix)

                # Also close all pyplot figures. It is expensive to do this every epoch
                plt.close("all")

            # Reset metrics every epoch
            vae_loss_metric.reset_states()
            kl_loss_metric.reset_states()
            generation_sharpness_loss_metric.reset_states()
            reconstruction_sharpness_loss_metric.reset_states()

            # tf.keras.backend.clear_session()

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
