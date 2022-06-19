from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def dense_block(
    neurons: int, activation: str = "relu", batch_norm: bool = True
) -> List[layers.Layer]:
    """
    Standard densely connected block
    """
    block = [layers.Dense(neurons)]
    if batch_norm:
        block.append(layers.BatchNormalization())
    if activation == "relu":
        block.append(layers.ReLU())
    return block


class DenseBlock(tf.keras.layers.Layer):
    def __init__(
        self, units, regularise=0.01, drop_prob=0.2, activation=tf.nn.relu, useBN=True
    ):
        super(DenseBlock, self).__init__()
        self.units = units
        self.activation = activation
        self.useBN = useBN

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
        if self.useBN:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h


def conv_2d_block(
    filters: int, activation: str = "relu", batch_norm: bool = True
) -> List[layers.Layer]:
    """
    Standard convolutional block
    """
    block = [
        layers.Conv2D(filters, kernel_size=5, strides=2, padding="same"),
    ]
    if batch_norm:
        block.append(layers.BatchNormalization())
    if activation == "relu":
        block.append(layers.ReLU())
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
        activation=tf.nn.relu,
        useBN=True,
    ):
        super(Conv2dBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.useBN = useBN

        if regularise == 0:
            self.conv = tf.keras.layers.Conv2D(
                filters, kernel_size, strides=strides, padding=padding, activation=None
            )
        else:
            self.conv = tf.keras.layers.Conv2D(
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
        if self.useBN:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h


def deconv_2d_block(
    filters: int, activation: str = "relu", batch_norm: bool = True
) -> List[layers.Layer]:
    """
    Standard densely connected block
    """
    block = [
        layers.Conv2DTranspose(filters, kernel_size=5, strides=2, padding="same"),
    ]
    if batch_norm:
        block.append(layers.BatchNormalization())
    if activation == "relu":
        block.append(layers.ReLU())
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
        useBN=True,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.useBN = useBN

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
        if self.useBN:
            h = self.bn(h, training)
        if self.activation:
            h = self.activation(h)
        h = self.drop(h, training)
        return h
