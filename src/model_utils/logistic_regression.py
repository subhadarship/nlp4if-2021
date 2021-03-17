import logging
from typing import List

from torch import nn

logger = logging.getLogger(__name__)


class MultitaskLogisticRegressionClassificationModel(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float, out_dims: List[int]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.clfs = nn.ModuleList([nn.Linear(hidden_size, out_dim) for out_dim in out_dims])

    def forward(self, x):
        # x = [batch size, max len]
        x = self.emb(x)
        # x = [batch size, max len, hidden size]
        x = x.sum(1)
        # x = [batch size, hidden size]
        x = self.dropout(x)
        # x = [batch size, hidden size]
        x = {f'q{idx + 1}': clf(x) for idx, clf in enumerate(self.clfs)}
        # x = {q1: [batch size, out dim], ...}
        return {
            'logits': x
        }
