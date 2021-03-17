import logging
import os

logger = logging.getLogger(__name__)


def init_logger(log_fpath):
    """Initialize logger"""
    if log_fpath is not None and os.path.dirname(log_fpath) != '':
        os.makedirs(os.path.dirname(log_fpath), exist_ok=True)

    logging.basicConfig(
        filename=log_fpath,  # if None, does not write to file
        filemode='a',  # default is 'a'
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
