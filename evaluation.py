import logging
from typing import List
import dotenv
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers.base import LightningLoggerBase


dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="eval_config.yaml")
def main(config: DictConfig):
    logger.info(config.pretty())

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import mlflow
    from hydra import utils
    from src.eval import eval
    from src.utils import model, config_utils

    config_utils.extras(config)

    log_dir = None
    # get the hydra logdir using the exp_id
    if config.get('exp_id'):
        client = mlflow.tracking.MlflowClient(
            tracking_uri=config.logger.mlflow.tracking_uri)
        data = client.get_run(config.get('exp_id')).to_dictionary()
        log_dir = data['data']['params']['hydra/log_dir']
    else:
        # TODO: Find a easier way to load a past configuration
        raise Exception(
            '`exp_id` must be defined in order to evaluate an existing experiment')

    # load the saved model and datamodule
    log_dir = utils.get_original_cwd() + '/' + log_dir
    model, datamodule, exp_config = model.load_experiment(log_dir)

    # instanciate mlflow and the trainer for the evaluation
    mlf_logger = utils.instantiate(
        config.logger.mlflow, experiment_name=exp_config.logger.mlflow.experiment_name)
    trainer = utils.instantiate(
        config.trainer, callbacks=[], logger=[mlf_logger], _convert_='partial'
    )

    return eval(config, model, trainer, datamodule)


if __name__ == "__main__":
    main()
