import logging
import random
from typing import List, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bert_data import BertInfodemicDataset
from .data import InfodemicDataset

logger = logging.getLogger(__name__)


class SMARTTOKDataLoader(DataLoader):
    """Dataloader for batching data by sorting based on sentence lengths. Batching is done such that the maximum number of tokens in a batch is fixed."""

    def __init__(self, dataset: Union[InfodemicDataset, BertInfodemicDataset], max_tokens: int, pad_idx: int,
                 shuffle: bool,
                 progress_bar: bool, device: torch.device):
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.shuffle = shuffle
        self.progress_bar = progress_bar
        self.device = device

        self.batches = []

        if self.shuffle:
            unsorted_lengths = [len(sample['text']) for sample in self.dataset]
            self.order = np.argsort(unsorted_lengths).tolist()
            # compute number of very long samples
            self.num_very_long_samples = 0
            for idx in range(self.num_samples - 1, -1, -1):
                if len(self.dataset[self.order[idx]]['text']) > self.max_tokens:
                    logger.warning(f"ignoring sample {self.order[idx]}"
                                   "(too big for specified max tokens in a batch)")
                    self.num_very_long_samples += 1
                else:
                    break
        else:
            self.order = list(range(self.num_samples))

    def init_epoch(self):
        # initialize empty list of batches
        self.batches = []
        if self.shuffle:
            # make a copy of the order list (deep copy)
            sorted_order = self.order.copy()
            # ignore the very long samples
            if self.num_very_long_samples > 0:
                sorted_order = sorted_order[:-self.num_very_long_samples]

            if self.progress_bar:
                bar = tqdm(total=len(sorted_order), desc='initialize epoch', unit=' samples', leave=False)

            while True:
                batch = self.create_batch(candidates=sorted_order, start_idx=len(sorted_order) - 1, direction=-1)
                if len(sorted_order) == len(batch):
                    # add the last batch
                    self.batches.append(batch[::-1])
                    if self.progress_bar:
                        # update progress bar
                        bar.update(len(batch))
                    break
                else:
                    # determine start index of the chunk
                    start_idx = random.randrange(0, len(sorted_order) - len(batch) + 1)
                    # create chunk starting from this index
                    batch = self.create_batch(candidates=sorted_order, start_idx=start_idx, direction=1)
                    # add chunk to self.batches
                    self.batches.append(batch)
                    # delete chunk
                    del sorted_order[start_idx: start_idx + len(batch)]
                    if self.progress_bar:
                        # update progress bar
                        bar.update(len(batch))

        else:
            self.create_batches()

    def create_batch(self, candidates: List[int], start_idx: int, direction: int) -> List[int]:
        """Create a batch from the given candidates list starting for the start_idx traversing to the left or right specified by direction. `direction` is -1 for left or 1 for right traversal"""
        if direction == 1:
            to_idx = len(candidates)
        elif direction == -1:
            to_idx = -1
        else:
            raise AssertionError(
                f'direction should be -1 for left traversal or 1 for right traversal, provided value is {direction}')
        batch = []
        batch_size_so_far, max_sent_tokens = 0, 0
        for cand_idx in range(start_idx, to_idx, direction):
            length = len(self.dataset[candidates[cand_idx]]['text'])
            batch_size_so_far += 1
            max_sent_tokens = max(max_sent_tokens, length)
            tot_tokens = batch_size_so_far * max_sent_tokens
            if tot_tokens <= self.max_tokens:
                batch.append(candidates[cand_idx])
            else:
                break
        return batch

    def create_batches(self):
        self.batches = []
        current_batch = []
        batch_size_so_far, max_sent_tokens = 0, 0

        bar = tqdm(self.order, desc='initialize epoch', unit=' batches', leave=False) if self.progress_bar else self.order
        for idx in bar:
            # three cases for self.dataset[idx]:
            # 1. it goes to the current batch
            # 2. it goes to the next batch
            # 3. it is omitted if it is too big for the specified max tokens

            length = len(self.dataset[idx]['text'])
            batch_size_so_far += 1
            max_sent_tokens = max(max_sent_tokens, length)
            tot_tokens = batch_size_so_far * max_sent_tokens

            if tot_tokens <= self.max_tokens:
                # add sample to current batch
                current_batch.append(idx)
            else:
                # wrap up current batch
                self.batches.append(current_batch)

                # create empty next batch
                current_batch = []
                batch_size_so_far, max_sent_tokens = 0, 0

                if length < self.max_tokens:
                    # add sample to created batch
                    current_batch.append(idx)
                    batch_size_so_far += 1
                    max_sent_tokens += length
                else:
                    # If this happens then there is one sample that is too big,
                    # just ignore it wth a warning
                    logger.warning(f"ignoring sample {idx}"
                                   "(too big for specified max tokens in a batch)")
        # add the last batch
        if len(current_batch) > 0:
            self.batches.append(current_batch)

    def __iter__(self):
        self.init_epoch()
        self.pos = 0
        return self

    def __len__(self):
        number_of_batches = len(self.batches)
        if number_of_batches == 0:
            logger.warning(f'initialize iterator before computing number of batches, returning 0')
        return number_of_batches

    def get_batch(self, pos):
        samples = [self.dataset[i] for i in self.batches[pos]]
        sentences = [sample['text'] for sample in samples]
        sentences = pad_sequence(sentences, batch_first=True, padding_value=self.pad_idx)
        return {
            'text': sentences.to(self.device),
            'labels': {
                f'q{idx + 1}': torch.cat([sample['labels'][f'q{idx + 1}'] for sample in samples]).to(self.device) for
                idx in range(7)
            }
        }

    def __next__(self):
        if self.pos >= len(self):
            raise StopIteration()
        batch = self.get_batch(self.pos)
        self.pos += 1
        return batch
