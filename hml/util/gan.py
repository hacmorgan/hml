#!/usr/bin/env python3


"""
Learning rate schedules
"""


__author__ = "Hamish Morgan"


from typing import Tuple


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

    This is determined based on both absolute loss and the change in loss since the last
    epoch. If after training for an epoch, the generator loss has passed the relevant
    absolute threshold, or the generator loss has not changed significantly, we switch
    to training the other network.

    n.b. these determinations are made on the generator's loss instead of the
    discriminator's, because the generator is purely trying to fool the discriminator,
    while the discriminator is also trying to learn to classify real images, which it
    tends to get pretty good at, meaning its loss trends downward, at least to a point.

    Args:
        should_train_generator: True if currently training generator
        should_train_discriminator: True if currently training discriminator
        this_generator_loss: Loss metric for generator after current epoch
        last_generator_loss: Loss metric for generator after previous epoch
        switch_training_loss_delta: If change in loss for this epoch is greater than
            this, the model is still learning
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
