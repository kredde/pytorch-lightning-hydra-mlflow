<div align="center">

# Pytorch Lightning Template using Hydra and MLFlow

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://mlflow.org/"><img alt="Tracking: MLFlow" src="https://img.shields.io/badge/tracking-mlflow-blue"></a>
</div>


## Quick start
1. Install dependencies `conda env create -f environment.yml -n envname`
2. Run an experiment using `python3 main.py +experiment=exp_name`
3. Check out the results of your runs using `mlflow ui`
4. Reload your experiment and execute evaluations by specifying the run id `exp_id: id` and running the evaluation pipeline `python3 evaluation.py +experiment=exp_name`

Train model with default configuration
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py +experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Other Repositories

<details>
<summary><b>Inspirations</b></summary>

This template was inspired by:

[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template),

[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),

[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),

[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),

[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),

[lucmos/nn-template](https://github.com/lucmos/nn-template).

</details>