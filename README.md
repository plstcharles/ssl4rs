<div align="center">

# SSL for Remote Sensing (SSL4RS) Sandbox

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Last Commit](https://img.shields.io/github/last-commit/plstcharles/ssl4rs/master)](https://github.com/plstcharles/ssl4rs)
[![License](https://img.shields.io/github/license/plstcharles/ssl4rs)](https://github.com/plstcharles/ssl4rs/blob/master/LICENSE)
[![tests](https://img.shields.io/github/actions/workflow/status/plstcharles/ssl4rs/test.yml)](https://github.com/plstcharles/ssl4rs/actions/workflows/test.yml)

</div>

## Description

A deep learning sandbox for Self-Supervised Learning (SSL) applications for Remote Sensing (RS).

This framework is primarily meant to help the prototyping of new models and data loaders. It relies
on [PyTorch](https://pytorch.org/get-started/locally/) in combination with
[Lightning](https://lightning.ai/), and is derived from the [Lightning-Hydra-Template
Project](https://github.com/ashleve/lightning-hydra-template).

The easiest way to use this framework is probably to clone it, add your own code inside its folder
structure, modify things as needed, and run your own experiments (derived from defaults/examples).
You can however also use it as a dependency if you are familiar with how Hydra configuration files
are handled.

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

This is a YAML-configuration-based Hydra project. Therefore, experiment configurations are defined
via a separate configuration file tree (`configs`; see the next section for more information).

The rest of the framework can be defined as follows:

```
<repository_root>
  ├── configs    => root directory for all configuration files; see next section
  │   └── ...
  ├── data       => suggested root directory for datasets (might not exist); can be a symlink
  │   ├── <some_dataset_directory>
  │   ├── <some_other_dataset_directory>
  │   └── ...
  ├── logs       => suggested root directory for outputs (might not exist); can be a symlink
  │   ├── comet
  │   ├── tensorboard
  │   ├── ...
  │   └── runs
  │       └── <some_experiment_name>
  │           ├── <some_run_name>
  │           │   ├── ckpts
  │           │   └── ...
  │           └── <some_other_run_name>
  │               └── ...
  ├── notebooks  => contains notebooks used for data analysis, visualization, and demonstrations
  │   └── ...
  ├── ssl4rs     => root directory for the framework's packages and modules
  │   ├── data                 => contains subpackages related to data loading
  │   │   ├── datamodules      => datamodules for different datasets
  │   │   ├── parsers          => dataset parsers used inside datamodules
  │   │   ├── repackagers      => contains dataset repackagers/converters
  │   │   └── transforms       => various data transformation classes/operations
  │   ├── models               => contains subpackages related to models/architectures
  │   │   └── components       => various basic components used for model building
  │   └── utils                => generic utility module for the whole framework
  └── tests     => contains unit tests for ssl4rs framework packages/modules
      └── ...
```

There are three 'entrypoint'-type scripts in the framework off which we can easily launch an
experiment. These are:

- `<repository_root>/train.py`: used to launch model training experiments; will load the
  configuration file at `configs/train.yaml` by default.
- `<repository_root>/test.py`: used to launch inference runs; will load the configuration file
  at `configs/test.yaml` by default.
- `<repository_root>/data_profiler.py`: used to profile datamodule creation, data loader
  initialization, and data sample loading; will load the configuration file at
  `configs/profiler.yaml` by default.
- `<repository_root>/model_profiler.py`: used to profile model training and validation epochs;
  will load the configuration file at `configs/profiler.yaml` by default.

## Configuration Files

When using Hydra, configuration files (or structures) are used to provide and log settings across
the entire application. For a tutorial on Hydra, see the
[official documentation](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).

In this framework, most of the already-existing configuration files provide default values for
settings across different categories. An experiment with a custom model, a custom dataset, custom
metrics, and/or other user-specified settings will likely rely on a new configuration file that
loads the default values and overrides some of them. Such experiment configuration files should be
placed in the `<repository_root>/configs/experiment/` directory.

The structure of all configuration directories is detailed below:

```
<repository_root>
└── configs       => root directory for all YAML configuration files
    ├── callbacks      => lists of commonly-used lightning callbacks
    ├── data           => definitions for datamodules and data loader settings
    ├── debug          => provides various overrides used to help debug experiments
    ├── experiment     => examples of experiment configs and potential user-provided ones
    ├── hparams_search => examples of hyperparameter search engine configurations
    ├── local          => contains machine-specific ("local") configuration overrides
    ├── logger         => settings for experiment logging tools such as tensorboard
    ├── model          => model architecture, optimization, and loss-specific settings
    ├── output         => output ("log") directory management settings
    ├── trainer        => lightning trainer settings
    └── utils          => generic framework-wide utility settings
```

For experiment configurations, these will typically override settings across the full scope
of the configuration tree, meaning that they will likely be defined with the `# @package _global_`
line. A good starting point on how to write such a configuration is to copy and modify one of the
examples, such as [this one](ssl4rs/configs/experiment/example_mnist_classif.yaml). This file can be
used to define overrides as well as new settings that may affect any aspect of an experiment
launched with the framework. Remember: to launch a training experiment for a file named
`some_new_experiment_config.yaml` in the `<repository_root>/configs/experiment/` directory, you
would run:

```bash
python train.py experiment=some_new_experiment_config
```

## Output files

The results of an experiment comes under the form of checkpoints, merged configuration files,
console logs, and any other artefact that your code may produce. By default, these will be saved
under the path defined by the `OUTPUT_ROOT` environment variable, under subdirectories named based
on experiment and run identifiers. A typical output folder structure following an experiment using
CSV and tensorboard loggers, launched for example with

```bash
python train.py experiment=example_mnist_classif_fast logger=tboard_and_csv
```

...will then look like this:

```
<OUTPUT_ROOT>
├── runs
│   └── mnist_with_micro_mlp      => experiment name
│       └── 20230329_163039       => run name
│           ├── ckpts             => where model checkpoints are saved
│           │   ├── e002_s005157.<...>.ckpt   => epoch 2, step 5157
│           │   └── last.ckpt                 => latest trainer checkpoint
│           ├── config.<...>.log  => backup config with fully-interpolated values
│           ├── console.log       => console log (concatenated across all launches)
│           ├── csv               => the lightning csv logger output dir
│           │   └── ...
│           ├── installed_pkgs.<...>.log  => list of installed python packages
│           └── runtime_tags.<...>.log    => dictionary of useful runtime info
└── tensorboard     => the lightning tensorboard logger output dir
    └── mnist_with_micro_mlp            => experiment name
        └── 20230329_163039_0           => run name
            ├── events.out.tfevents.<...>
            └── ...
```

## Other Notes

For more info on the usage of the config files and hydra/Lightning tips+tricks, see the
[original template repository](https://github.com/ashleve/lightning-hydra-template).
