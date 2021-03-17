import logging
import os
from typing import List

import pandas as pd

logging.getLogger(__name__)


def load_dataframe(fpath: str) -> pd.DataFrame:
    """Prepare dataframe"""
    assert os.path.isfile(fpath)
    return pd.read_csv(fpath, sep='\t', encoding='utf-8', na_filter=False)


def concat_dataframe(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate dataframes"""
    return pd.concat(dfs)


def sample_dataframe(df: pd.DataFrame, num_samples: int, random_seed: int) -> pd.DataFrame:
    """Select a fraction of rows dataframe"""
    return df.sample(n=num_samples, random_state=random_seed)
