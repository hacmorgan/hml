# hml
Machine learning models and tools


# Installation
`hml` is installed as a Python package using `pip`:

    git clone git@github.com:hacmorgan/hml
    python3 -m pip install --upgrade hml


# Usage
When a model is first trained, an experiment directory is created. In addition to storing training logs and checkpoints, and any progress outputs (e.g. for a generative model), `hml` will copy the model architecture module(s) there, such that they can be retrieved later for continuing to train or inference/generation, even if the latest version of the architecture has been changed.
