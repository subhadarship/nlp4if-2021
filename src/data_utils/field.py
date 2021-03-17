import logging
from collections import Counter
from typing import List, Union

from tqdm import tqdm

from .nltk_tokenizer import NLTKTokenizer
from .tweet_tokenizer import TweetTokenizerNormalizer

logger = logging.getLogger(__name__)


class Field(object):
    """Field that handles tokenization, vocab and itos and stoi"""

    def __init__(self, tokenizer: Union[NLTKTokenizer, TweetTokenizerNormalizer], max_vocab: Union[int, None],
                 min_freq: int,
                 pad_token: Union[str, None],
                 unk_token: Union[str, None],
                 sos_token: Union[str, None],
                 eos_token: Union[str, None]):
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.freqs = None
        self.itos = None
        self.stoi = None
        self.excluded_tokens = None
        self.vocab_size = None

    def build_vocab(self, text: List[str]):
        """Build vocabulary"""

        # first initialize vocab, itos, stoi, tokens to exclude
        self.freqs = Counter()
        self.itos = [self.pad_token, self.unk_token]
        self.stoi = {self.pad_token: 0, self.unk_token: 1}

        # track offset
        offset = 2  # for pad and unk tokens

        if self.sos_token is not None:
            self.itos.append(self.sos_token)
            self.stoi[self.sos_token] = len(self.itos) - 1
            offset += 1
        if self.eos_token is not None:
            self.itos.append(self.eos_token)
            self.stoi[self.eos_token] = len(self.itos) - 1
            offset += 1
        self.excluded_tokens = []

        for sentence in tqdm(text, desc='build vocab', unit=' sentences'):
            tokens = self.tokenizer.tokenize(sentence)
            self.freqs.update(tokens)

        count = 0
        for idx, (token, freq) in enumerate(self.freqs.most_common()):
            if freq >= self.min_freq and (self.max_vocab is None or count < self.max_vocab):
                self.itos.append(token)
                self.stoi[token] = idx + offset
                count += 1
            else:
                self.excluded_tokens.append(token)

        logger.info(
            f'{len(self.freqs) - len(self.excluded_tokens)} '
            f'({100 * (len(self.freqs) - len(self.excluded_tokens)) / len(self.freqs):0.2f}%) '
            f'tokens out of {len(self.freqs)} tokens are kept in vocabulary'
        )

        # pop tokens with frequency less than self.min_freq
        for token_to_pop in self.excluded_tokens:
            self.freqs.pop(token_to_pop)

        # update vocab size
        self.vocab_size = len(self.itos)

    def preprocess(self, s: str):
        tokens = self.tokenizer.tokenize(s)
        out = []
        for token in tokens:
            out.append(token) if token in self.freqs else out.append(self.unk_token)
        if self.sos_token is not None:
            # add sos token to left
            out = [self.sos_token] + out
        if self.eos_token is not None:
            # add eos token to right
            out = out + [self.eos_token]
        return out

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self.stoi[token] for token in tokens]


class LabelField(object):
    """Field for a label. Handles vocab, itos, stoi"""

    def __init__(self, labels: List[str]):
        self.itos = labels
        self.stoi = {label: idx for idx, label in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
