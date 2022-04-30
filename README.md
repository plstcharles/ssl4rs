<div align="center">

# SSL for Remote Sensing (SSL4RS) Sandbox

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-LH--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

TODO

## How to run

Install dependencies

```bash
# clone project
git clone <URL_TO_REPO>
cd your-repo-name

# create conda environment
conda create -f environment.yaml
conda activate ssl4rs
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 data.batch_size=64
```

For more info on the usage of the config files and hydra/PyTorch-Lightning tips+tricks, see the
[original template repository](https://github.com/ashleve/lightning-hydra-template).
