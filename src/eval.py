import logging

from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything

from omegaconf import DictConfig

from src.utils import config_utils

log = logging.getLogger(__name__)


def eval(config: DictConfig, model: LightningModule, trainer: Trainer, datamodule: LightningDataModule) -> Optional[float]:
    """Contains the evaluation pipeline.

    Uses the configuration to execute the evaluation pipeline on a given model.

    args:
        config (DictConfig): Configuration composed by Hydra.
        model (LightningModule): The model that is evaluated
        trainer (Trainer)
        datamodule (LightningDataModule)
    """

    if 'seed' in config:
        seed_everything(config.seed)

    # Send some parameters from config to all lightning loggers
    log.info('Logging hyperparameters!')
    config_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=[],
        logger=trainer.logger,
    )

    # add your evaluation logic here
