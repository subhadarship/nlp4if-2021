import logging
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from .data import COLUMN_NAMES
from .field import LabelField

logger = logging.getLogger(__name__)


class BertInfodemicDataset(Dataset):
    """Bert Infodemic dataset"""

    def __init__(self, df: pd.DataFrame, bert_tokenizer: BertTokenizer, label_fields: List[LabelField]):

        self.bert_tokenizer = bert_tokenizer
        self.label_fields = label_fields

        self.all_original_sentences = df[COLUMN_NAMES[0]].astype(str).tolist()
        self.labels = {
            f'q{idx + 1}': df[COLUMN_NAMES[idx + 1]].astype(str) for idx in range(7)
        }

        self.all_sent_ids = []
        self.all_label_ids = []
        for sample_idx, sentence in enumerate(
                tqdm(self.all_original_sentences, desc='prepare bert data', unit=' samples')):
            ids = self.bert_tokenizer.encode(sentence)  # [CLS idx, ..., SEP idx]
            if len(ids) > self.bert_tokenizer.model_max_length:
                logger.warning(
                    f'trimming sentence {sample_idx} of length {len(ids)} to {self.bert_tokenizer.model_max_length} tokens '
                    f'(trimmed tokens include {self.bert_tokenizer.cls_token} and {self.bert_tokenizer.sep_token} tokens)'
                )
                ids = ids[:self.bert_tokenizer.model_max_length - 1] + [self.bert_tokenizer.sep_token_id]

            self.all_sent_ids.append(torch.LongTensor(ids))
            label_ids = {}
            for idx in range(7):
                label_ids[f'q{idx + 1}'] = torch.LongTensor(
                    [self.label_fields[idx].stoi[self.labels[f'q{idx + 1}'][sample_idx]]])
            self.all_label_ids.append(label_ids)

    def __getitem__(self, idx):
        return {
            'text': self.all_sent_ids[idx],
            'labels': self.all_label_ids[idx],
            'orig': self.all_original_sentences[idx],
            'orig_labels': ' '.join([self.labels[f'q{label_idx + 1}'][idx] for label_idx in range(7)]),
        }

    def __len__(self):
        return len(self.all_sent_ids)
