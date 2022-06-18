from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from hml.architectures.convolutional.blocks import (
    Deconv2dBlock,
    DenseBlock,
    deconv_2d_block,
    dense_block,
)


def decoder(latent_dim: int) -> tf.keras.Sequential:
    """
    An encoder based on the DCGAN discriminator
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    architecture = tf.keras.Sequential(
        [
            # Latent input
            layers.InputLayer(input_shape=(latent_dim,)),
            *dense_block(neurons=9 * 16 * 128),
            layers.Reshape((9, 16, 128)),
            # # Output shape: (9, 16, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (18, 32, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (36, 64, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (72, 128, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (144, 256, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (288, 512, 128)
            *deconv_2d_block(filters=128),
            # Output shape: (576, 1024, 128)
            *deconv_2d_block(filters=3, batch_norm=False),
            # Output shape: (1152, 2048, 128)
        ]
    )
    architecture.build()
    architecture.summary()
    return architecture


class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.input_ = layers.InputLayer(input_shape=latent_dim)
        self.dense = DenseBlock(units=(9 * 16 * 128))
        self.reshape = layers.Reshape((9, 16, 128))
        self.conv1 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 18, 32
        self.conv2 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 36, 64
        self.conv3 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 72, 128
        self.conv4 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 144, 256
        self.conv5 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 288, 512
        self.conv6 = Deconv2dBlock(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
        )
        # Output shape: 576, 1024
        self.conv7 = Deconv2dBlock(
            filters=3,
            kernel_size=5,
            strides=(2, 2),
            activation=tf.nn.sigmoid,
            drop_prob=0,
            useBN=False,
        )
        # Output shape: 1152, 2048

    def call(self, inputs, training=False):
        h0 = self.input_(inputs)
        hd = self.dense(h0, training=training)
        hr = self.reshape(hd, training=training)
        h1 = self.conv1(hr, training=training)
        h2 = self.conv2(h1, training=training)
        h3 = self.conv3(h2, training=training)
        h4 = self.conv4(h3, training=training)
        h5 = self.conv5(h4, training=training)
        h6 = self.conv6(h5, training=training)
        h7 = self.conv7(h6, training=training)
        return h7
