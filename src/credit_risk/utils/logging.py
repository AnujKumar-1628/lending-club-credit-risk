import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    """

    logger = logging.getLogger(name)

    # Prevent duplicate logs in notebooks / repeated imports
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
