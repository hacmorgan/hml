#!/usr/bin/env python3


"""
Learning rate schedules
"""


__author__ = "Hamish Morgan"


import tensorflow as tf


class WingRampLRS(tf.keras.optimizers.schedules.LearningRateSchedule):
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
    ) -> "WingRampLRS":
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
