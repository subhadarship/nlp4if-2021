import logging
import os
from typing import Union

import pandas as pd

from .dataframe import load_dataframe, sample_dataframe, concat_dataframes

logger = logging.getLogger(__name__)


def load_data(langs_with_num_samples: str, data_dir: str, split_name: str, random_seed: int) \
        -> Union[pd.DataFrame, None]:
    """Load data"""

    if data_dir is None:
        return None

    langs_numsamples = langs_with_num_samples.split(',')
    langs = [item.split('_')[0] for item in langs_numsamples]
    num_samples = [item.split('_')[-1] if len(item.split('_')) > 1 else '' for item in langs_numsamples]
    num_samples = [-1 if item == 'all' or item == '' else int(item) for item in num_samples]
    dfs = []
    for lang, num_s in zip(langs, num_samples):
        df = load_dataframe(fpath=os.path.join(data_dir, f'{split_name}.{lang}.tsv'))
        total_samples = len(df)
        if 0 < num_s < total_samples:
            picked = num_s
            df = sample_dataframe(df, num_samples=num_s, random_seed=random_seed)
        else:
            picked = total_samples
        logger.info(
            f'considered {picked} ({100 * picked / total_samples:0.2f} %) samples out of {total_samples} total samples in {os.path.join(data_dir, f"{split_name}.{lang}.tsv")}')
        dfs.append(df)
    return concat_dataframes(dfs)
