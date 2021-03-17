import logging

logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count number of trainable parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
