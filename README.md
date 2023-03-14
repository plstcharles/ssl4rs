<div align="center">

# SSL for Remote Sensing (SSL4RS) Sandbox

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/%22)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![LH-Template](https://img.shields.io/badge/-LH--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![tests](https://github.com/plstcharles/ssl4rs/actions/workflows/test.yml/badge.svg)](https://github.com/plstcharles/ssl4rs/actions/workflows/test.yml)

</div>

## Description

A deep learning sandbox for Self-Supervised Learning (SSL) applications for Remote Sensing (RS).

This framework is primarily meant to help the prototyping of new models and data loaders. It relies
on [PyTorch](https://pytorch.org/get-started/locally/) in combination with
[PyTorch Lightning](https://pytorchlightning.ai/), and is derived from the [Lightning-Hydra-Template
Project](https://github.com/ashleve/lightning-hydra-template).

## How to run an experiment

First, install the framework and its dependencies:

```bash
# clone project
git clone https://github.com/plstcharles/ssl4rs
cd ssl4rs

# create conda environment
conda create -n ssl4rs python=3.10 pip
conda activate ssl4rs
pip install -r requirements.txt
```

Next, create a copy of the `.env.template` file, rename it to `.env`, and modify its content so
that at least all mandatory variables are filled. These include:

- `DATA_ROOT`: path to the root directory where all datasets are located. It will be internally
  used via Hydra/OmegaConf through the `utils.data_root_dir` config key. All datamodules that are
  implemented in the framework will likely define their root directory based on this location.
- `OUTPUT_ROOT`: path to the root directory where all outputs (logs, checkpoints, images, ...) will
  be written. It will be internally used via Hydra/OmegaConf through the `utils.output_root_dir`
  config key. It is at that location where experiment and run directories will be created.

Note that this file is machine-specific, and it may contain secrets and API keys. Therefore, it will
always be ignored by version control (due to the `.gitignore` filters), and you should be careful
about logging its contents or printing it inside a script to avoid credential leaks.

Finally, launch an experiment using an existing config file, or create a new one:

```bash
python train.py experiment=example_mnist_classif_fast
# or
python test.py experiment=example_mnist_classif_fast ckpt_path=<PATH_TO_AN_EXISTING_CHECKPOINT>
```

Note that since the entrypoints are Hydra-based, you can override parameters from the command line:

```bash
python train.py experiment=example_mnist_classif_fast trainer.max_epochs=3
```

The experiment configuration files provide the main location from where settings should be modified
to run particular experiments. New experiments can be defined by copying and modifying existing
files. For more information on these files, see the [relevant section](#configuration-files).

## Framework Structure

TODO! @@@@@@

## Configuration Files

TODO! @@@@@@

## Other Notes

For more info on the usage of the config files and hydra/PyTorch-Lightning tips+tricks, see the
[original template repository](https://github.com/ashleve/lightning-hydra-template).
