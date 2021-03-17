import logging

import torch
from transformers import BertConfig

from .bert import BERT, MultitaskBertClassificationModel

logger = logging.getLogger(__name__)


def load_bert_multitask_classification_model(
        model_name: str,
        model_hyperparams_dict: dict,
        data_dict: dict,
        freeze: bool,
        device: torch.device,
):
    """Load BERT multitask classification model"""
    TARGET_SIZES = [LABEL.vocab_size for LABEL in data_dict['LABELS']]

    FC_DIM = model_hyperparams_dict['FC_DIM']

    # src pad idx
    src_pad_idx = data_dict['TOKENIZER'].pad_token_id
    # config
    config = BertConfig.from_pretrained(model_name)
    # bert as encoder
    encoder = BERT.from_pretrained(model_name, config=config)

    # freeze weights of BERT (optionally)
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    # multitask text classification model
    model = MultitaskBertClassificationModel(
        encoder=encoder,
        src_pad_idx=src_pad_idx,
        out_dims=TARGET_SIZES,
        fc_dim=FC_DIM,
    ).to(device)

    return model
