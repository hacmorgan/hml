#!/usr/bin/env python3
from distutils.core import setup

setup(
    name="hml",
    version="0.1.0",
    description="Machine learning models, data pipelines, tools, etc",
    author="Hamish Morgan",
    author_email="ham430@gmail.com",
    url="http://github.com/hacmorgan/hml",
    packages=[
        "hml.architectures.convolutional.autoencoders",
        "hml.architectures.convolutional.discriminators",
        "hml.architectures.convolutional.encoders",
        "hml.architectures.convolutional.decoders",
        "hml.architectures.convolutional.generators",
        "hml.data_pipelines.unsupervised",
        "hml.layers",
        "hml.models",
        "hml.util",
    ],
    scripts=[
        "hml/models/vae.py",
        "hml/models/flood_sharpvae.py",
        "hml/models/pixel_art.py",
        "hml/models/pixel_art_ae.py",
        "hml/models/pixel_art_vae.py",
        "hml/models/pixel_art_avae.py",
        "hml/models/sample_gan.py",
        "hml/run_scripts/train-collage-vae",
    ],
    install_requires=[
        "GitPython",
        "opencv-python",
        "matplotlib",
        # "pygobject",
        # "pycairo",
        "tensorflow",
        "tensorflow_addons",
        "tensorflow_datasets",
        "tensorflow_gan",
        "tensorflow_io",
    ],
)
