import functools
import logging
import sys


@functools.lru_cache()
def setup_logger(name=__name__, verbose=True):
    logger = logging.getLogger(name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    level = logging.INFO if verbose else logging.ERROR

    logger.setLevel(level)

    return logger


logger = setup_logger()
