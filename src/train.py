import logging

from typing import List, Optional

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils import config_utils

log = logging.getLogger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline.

    Uses the config to instantiate the dataset, model and trainer.

    args:
        config (DictConfig): Configuration composed by Hydra.
    """

    if 'seed' in config:
        seed_everything(config.seed)

    # setup data module
    log.info(f'Instantiating datamodule <{config.datamodule._target_}>')
    datamodule: LightningDataModule = instantiate(config.datamodule)

    # setup model
    log.info(f'Instantiating model <{config.model._target_}>')
    model: LightningModule = instantiate(config.model)

    # setup callbacks
    callbacks: List[Callback] = []
    if 'callbacks' in config:
        for _, cb_conf in config['callbacks'].items():
            if '_target_' in cb_conf:
                log.info(f'Instantiating callback <{cb_conf._target_}>')
                callbacks.append(instantiate(cb_conf))

    # setup logger
    logger: List[LightningLoggerBase] = []
    if 'logger' in config:
        for _, lg_conf in config['logger'].items():
            if '_target_' in lg_conf:
                log.info(f'Instantiating logger <{lg_conf._target_}>')
                pl_logger = instantiate(lg_conf)
                logger.append(pl_logger)
            if lg_conf['_target_'] == 'pytorch_lightning.loggers.MLFlowLogger':
                mlf_logger = pl_logger

    # Init Lightning trainer
    log.info(f'Instantiating trainer <{config.trainer._target_}>')
    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_='partial'
    )

    # Send some parameters from config to all lightning loggers
    log.info('Logging hyperparameters!')
    config_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    mlf_logger.experiment.log_param(
        mlf_logger._run_id, "ml-flow/run_id", mlf_logger._run_id)

    # Train the model
    log.info('Training')
    trainer.fit(model=model, datamodule=datamodule)

    # evaluate model on test set after training
    if not config.trainer.get('fast_dev_run'):
        log.info('Testing')
        result = trainer.test()
        log.info('TEST RESULT: ' +
                 ' '.join([f'{key}: {result[0][key]}' for key in result[0].keys()]))

    # Print path to best checkpoint
    log.info(
        f'Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}')
