import logging

import nltk

logger = logging.getLogger(__name__)


class NLTKTokenizer(object):
    """NLTK tokenizer class"""

    def __init__(self):
        pass

    @staticmethod
    def tokenize(s: str):
        return nltk.word_tokenize(s)
