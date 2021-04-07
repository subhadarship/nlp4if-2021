import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(inp_fpath, dev_fraction: float, out_train_fpath: str, out_dev_fpath: str) -> None:
    """Split data into train and dev"""
    df = pd.read_csv(inp_fpath, sep='\t', encoding='utf-8', na_filter=False)

    # split
    train_df, dev_df = train_test_split(df, test_size=dev_fraction, random_state=123)

    # sizes
    print(f'looking at {fpath}..')
    print('\toriginal size:', len(df))
    print(df.isnull().sum(axis=0))
    print('\ttrain size:', len(train_df))
    print(train_df.isnull().sum(axis=0))
    print('\tdev size:', len(dev_df))
    print(dev_df.isnull().sum(axis=0))

    # write to files
    train_df.to_csv(out_train_fpath, sep='\t', encoding='utf-8', index=False)
    dev_df.to_csv(out_dev_fpath, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    DATA_PATH_ENGLISH = os.path.join('../data/english/v1/v1/covid19_disinfo_binary_english_train.tsv')
    DATA_PATH_ARABIC = os.path.join('../data/arabic/v1/v1/covid19_disinfo_binary_arabic_train.tsv')
    assert os.path.isfile(DATA_PATH_ENGLISH)
    assert os.path.isfile(DATA_PATH_ARABIC)

    for fpath in DATA_PATH_ENGLISH, DATA_PATH_ARABIC:
        split_data(
            inp_fpath=fpath,
            dev_fraction=1 / 6,
            out_train_fpath=os.path.join(os.path.dirname(fpath), 'train.tsv'),
            out_dev_fpath=os.path.join(os.path.dirname(fpath), 'dev.tsv'),
        )
