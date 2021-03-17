import logging
from typing import Union

import pandas as pd
import torch
from torch.utils.data import random_split

from .data import InfodemicDataset
from .dataloader import SMARTTOKDataLoader
from .field import Field, LabelField
from .length import compute_max_len
from .nltk_tokenizer import NLTKTokenizer
from .oov import compute_num_unk
from .tweet_tokenizer import TweetTokenizerNormalizer

logger = logging.getLogger(__name__)

DEBUG = False


def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: Union[pd.DataFrame, None],
                    tokenization_name: str, sos_token: Union[str, None], max_len: Union[int, None],
                    batch_size: int,
                    max_vocab_size: Union[int, None],
                    device: torch.device):
    """Preprocess data. Create fields, datasets and data iterators"""

    logger.info(f'tokenization: {tokenization_name}')
    if tokenization_name == 'nltk':
        tokenizer = NLTKTokenizer()
    elif tokenization_name == 'tweet':
        tokenizer = TweetTokenizerNormalizer()
    else:
        raise NotImplementedError

    # define fields
    TEXT = Field(tokenizer, max_vocab=max_vocab_size, min_freq=1,
                 pad_token='<pad>',
                 unk_token='<unk>',
                 sos_token=sos_token,
                 eos_token=None, )
    LABELS = [
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
        LabelField(['yes', 'no', 'nan']),
    ]

    # create datasets
    train_data = InfodemicDataset(df=train_df, text_field=TEXT, label_fields=LABELS, build_vocab=True, max_len=max_len)
    val_data = InfodemicDataset(df=val_df, text_field=TEXT, label_fields=LABELS, build_vocab=False, max_len=max_len)
    test_data = InfodemicDataset(df=test_df, text_field=TEXT, label_fields=LABELS,
                                 build_vocab=False, max_len=max_len) if test_df is not None else None

    # create data loaders
    train_iter = SMARTTOKDataLoader(dataset=train_data,
                                    max_tokens=batch_size,
                                    pad_idx=TEXT.stoi[TEXT.pad_token],
                                    shuffle=True,
                                    progress_bar=True,
                                    device=device)
    val_iter = SMARTTOKDataLoader(dataset=val_data,
                                  max_tokens=batch_size,
                                  pad_idx=TEXT.stoi[TEXT.pad_token],
                                  shuffle=False,
                                  progress_bar=True,
                                  device=device)
    test_iter = SMARTTOKDataLoader(dataset=test_data,
                                   max_tokens=batch_size,
                                   pad_idx=TEXT.stoi[TEXT.pad_token],
                                   shuffle=False,
                                   progress_bar=True,
                                   device=device) if test_data is not None else None

    # debug helper
    train_data = random_split(train_data, [100, len(train_data) - 100])[0] if DEBUG else train_data

    # log data stats
    logger.info(f'num train samples: {len(train_data)}')
    logger.info(f'num val samples: {len(val_data)}')
    logger.info(f'num test samples: {len(test_data) if test_data is not None else None}')

    for split_name, dataset in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        if dataset is None:
            continue
        max_len_src = compute_max_len(dataset)
        logger.info(f'{split_name} sentence max len: {max_len_src}')

    for split_name, dataset in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        if dataset is None:
            continue
        unk_and_total = compute_num_unk(dataset=dataset,
                                        unk_idx=TEXT.stoi[TEXT.unk_token])
        logger.info(
            f'{split_name} OOV: {unk_and_total["running_unk_tokens"]} ({100 * unk_and_total["running_unk_tokens"] / unk_and_total["running_total_tokens"]:0.2f}%)'
            f' out of {unk_and_total["running_total_tokens"]} running tokens are OOV')

    # show the first 5 tokenized sentences in train data and their labels
    logger.info(f'look at some train samples 👀')
    for idx in range(5):
        logger.info(
            f'sample idx: {idx}, '
            f'original text: {train_data[idx]["orig"]}, '
            f'preprocessed text: {train_data[idx]["prepro"]}, '
            f'text ids: {train_data[idx]["text"].tolist()}, '
            f'original labels: {train_data[idx]["orig_labels"]}, '
            f'label ids: {str([train_data[idx]["labels"][f"q{label_idx + 1}"].tolist() for label_idx in range(7)])}'
        )

    # estimate the number of batches in an epoch
    for batch_idx, _ in enumerate(train_iter):
        pass
    logger.info(f'there are nearly {batch_idx + 1} batches in an epoch')

    data_dict = {
        'TEXT': TEXT,
        'LABELS': LABELS,
        'TOKENIZER': tokenizer,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_iter': train_iter,
        'val_iter': val_iter,
        'test_iter': test_iter,
    }

    return data_dict
