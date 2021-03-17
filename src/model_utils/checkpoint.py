import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(model_dir, device):
    """Load checkpoint"""
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    return checkpoint_dict


def save_checkpoint(checkpoint_dict, model_dir):
    """Save checkpoint"""
    # create save dir
    os.makedirs(model_dir, exist_ok=True)
    torch.save(checkpoint_dict, os.path.join(model_dir, 'checkpoint.pt'))
