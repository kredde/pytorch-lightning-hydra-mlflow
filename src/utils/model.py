from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf


def load_experiment(path: str, checkpoint: str = 'last.ckpt'):
    """
      Loads an existing model and its dataloader.

      args:
        path (string): The path to the log folder
        checkpoint (string): the name of the checkpoint file
    """
    # load conf
    config: DictConfig = OmegaConf.load(path + '/.hydra/config.yaml')

    # reinitialize model and datamodule
    model = get_class(config.model._target_)
    datamodule: LightningDataModule = instantiate(config.datamodule)
    datamodule.setup()

    if "seed" in config:
        seed_everything(config.seed)

    model = model.load_from_checkpoint(path + '/checkpoints/' + checkpoint)

    return model, datamodule, config
