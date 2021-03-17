import logging
from typing import Union

from tqdm import tqdm

from .bert_data import BertInfodemicDataset
from .data import InfodemicDataset

logger = logging.getLogger(__name__)


def compute_max_len(dataset: Union[InfodemicDataset, BertInfodemicDataset]) -> int:
    """Compute maximum length of sentence in dataset"""
    max_len = -1
    for sample in tqdm(dataset, desc='compute max length', unit=' samples'):
        max_len = max(max_len, len(sample['text']))
    return max_len
