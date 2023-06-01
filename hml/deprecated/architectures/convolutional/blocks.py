from typing import Callable, List, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers


def dense_block(
    neurons: int,
    activation: Union[str, Callable] = tf.nn.relu,
    batch_norm: bool = True,
    bias: bool = True,
    drop_prob: float = 0.3,
    kernel_initializer: Union[
        str, tf.keras.initializers.Initializer
    ] = "glorot_uniform",
) -> List[layers.Layer]:
    """
    Standard densely connected block
    """
    block = [
        layers.Dense(neurons, kernel_initializer=kernel_initializer, use_bias=bias)
    ]
    if batch_norm:
        block.append(layers.BatchNormalization())
    if activation:
        block.append(layers.Activation(activation))
    if drop_prob:
        block.append(layers.Dropout(drop_prob))
    return block


class DenseBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        regularise=0.01,
        drop_prob=0.2,
        activation=tf.nn.relu,
        batch_norm=True,
    ):
        super(DenseBlock, self).__init__()
        self.units = units
        self.activation = activation
        self.batch_norm = batch_norm

        if regularise == 0:
            self.dense = tf.keras.layers.Dense(units, activation=None)
        else:
            self.dense = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L1(regularise),
                activity_regularizer=tf.keras.regularizers.L2(regularise),
            )

        self.drop = tf.keras.layers.Dropout(drop_prob)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):

        h = self.dense(inputs)
        if self.batch_norm:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h


def conv_2d_block(
    filters: int,
    strides: int = 2,
    activation: Union[str, Callable] = tf.nn.relu,
    batch_norm: bool = True,
    bias: bool = True,
    drop_prob: float = 0.3,
    kernel_initializer: Union[
        str, tf.keras.initializers.Initializer
    ] = "glorot_uniform",
) -> List[layers.Layer]:
    """
    Standard convolutional block
    """
    block = [
        layers.Conv2D(
            filters,
            kernel_size=5,
            strides=strides,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=bias,
        ),
    ]
    if batch_norm:
        block.append(layers.BatchNormalization())
    if activation:
        block.append(layers.Activation(activation))
    if drop_prob:
        block.append(layers.Dropout(drop_prob))
    return block


class Conv2dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        regularise=0.01,
        drop_prob=0.2,
        kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform",
        activation=tf.nn.relu,
        batch_norm=True,
        conv_type: str = "standard",
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_norm = batch_norm

        args = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": padding,
            "activation": None,
            "kernel_initializer": kernel_initializer,
        }
        if regularise != 0:
            args.update(
                {
                    "kernel_regularizer": tf.keras.regularizers.L1(regularise),
                    "activity_regularizer": tf.keras.regularizers.L2(regularise),
                    "kernel_initializer": kernel_initializer,
                }
            )

        if conv_type == "standard":
            self.conv = tf.keras.layers.Conv2D(**args)

        elif conv_type == "separable":
            self.conv = tf.keras.layers.SeparableConvolution2D(**args)

        else:
            raise ValueError(f"Unknown {conv_type=}")

        self.drop = tf.keras.layers.Dropout(drop_prob)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):

        h = self.conv(inputs)
        if self.batch_norm:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h


def deconv_2d_block(
    filters: int,
    strides: int = 2,
    activation: Union[str, Callable] = tf.nn.relu,
    batch_norm: bool = True,
    bias: bool = True,
    drop_prob: float = 0.3,
    kernel_initializer: Union[
        str, tf.keras.initializers.Initializer
    ] = "glorot_uniform",
) -> List[layers.Layer]:
    """
    Standard densely connected block - functional API
    """
    block = [
        layers.Conv2DTranspose(
            filters,
            kernel_size=5,
            strides=strides,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=bias,
        ),
    ]
    if activation:
        block.append(layers.Activation(activation))
    if batch_norm:
        block.append(layers.BatchNormalization())
    if drop_prob:
        block.append(layers.Dropout(drop_prob))
    return block


class Deconv2dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        regularise=0.01,
        drop_prob=0.2,
        activation=tf.nn.relu,
        batch_norm=True,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch = batch_norm

        if regularise == 0:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size, strides=strides, padding=padding, activation=None
            )
        else:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L1(regularise),
                activity_regularizer=tf.keras.regularizers.L2(regularise),
            )

        self.drop = tf.keras.layers.Dropout(drop_prob)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        h = self.conv(inputs)
        if self.batch_norm:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h
