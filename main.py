import logging
import dotenv
import hydra
from omegaconf import DictConfig


dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    logger.info(config.pretty())

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import config_utils

    config_utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
