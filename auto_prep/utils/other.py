import os

import matplotlib.pyplot as plt

from .config import config
from .logging_config import setup_logger

logger = setup_logger(__name__)


def save_chart(name: str, *args, **kwargs) -> str:
    """
    Saves chart to directory specified in config.

    Args:
        name (str) - Chart name with extension.
    Returs:
        path (str) - Path where chart has been saved.
    """
    path = os.path.join(config.charts_dir, name)
    logger.debug(f"Saving to {path}...")
    plt.savefig(os.path.join(config.charts_dir, name), *args, **kwargs)
    plt.close()
    return path
