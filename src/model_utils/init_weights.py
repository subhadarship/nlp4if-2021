import logging

from torch import nn

logger = logging.getLogger(__name__)


def initialize_weights(m):
    """Xavier initialization of weights"""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
