import logging

import torch

from .transformer_enc import Encoder, EncoderLayer, SelfAttentionLayer, PositionwiseFeedforwardLayer
from .transformer_enc import MultitaskTransformerEncoderClassificationModel

logger = logging.getLogger(__name__)


def load_transformer_enc_multitask_classification_model(
        model_hyperparams_dict: dict,
        data_dict: dict,
        device: torch.device,
):
    """Load transformer encoder multitask classification model"""
    TARGET_SIZES = [LABEL.vocab_size for LABEL in data_dict['LABELS']]

    TEXT_VOCAB_SIZE = data_dict['TEXT'].vocab_size
    HID_DIM = model_hyperparams_dict['HID_DIM']
    ENC_LAYERS = model_hyperparams_dict['ENC_LAYERS']
    ENC_HEADS = model_hyperparams_dict['ENC_HEADS']
    ENC_PF_DIM = model_hyperparams_dict['ENC_PF_DIM']
    ENC_DROPOUT = model_hyperparams_dict['ENC_DROPOUT']
    FC_DIM = model_hyperparams_dict['FC_DIM']

    src_pad_idx = data_dict['TEXT'].stoi[data_dict['TEXT'].pad_token]

    # encoder model
    encoder = Encoder(
        src_vocab_size=TEXT_VOCAB_SIZE,
        hid_dim=HID_DIM,
        n_layers=ENC_LAYERS,
        n_heads=ENC_HEADS,
        pf_dim=ENC_PF_DIM,
        encoder_layer=EncoderLayer,
        self_attention_layer=SelfAttentionLayer,
        positionwise_feedforward_layer=PositionwiseFeedforwardLayer,
        dropout=ENC_DROPOUT,
        device=device,
    )

    # multitask text classification model
    model = MultitaskTransformerEncoderClassificationModel(
        encoder=encoder,
        src_pad_idx=src_pad_idx,
        out_dims=TARGET_SIZES,
        fc_dim=FC_DIM,
    ).to(device)

    return model
