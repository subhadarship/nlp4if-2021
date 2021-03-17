import logging

import torch

from .logistic_regression import MultitaskLogisticRegressionClassificationModel

logger = logging.getLogger(__name__)


def load_logistic_regression_multitask_classification_model(
        model_hyperparams_dict: dict,
        data_dict: dict,
        device: torch.device,
):
    """Load logistic regression multitask classification model"""
    TARGET_SIZES = [LABEL.vocab_size for LABEL in data_dict['LABELS']]
    TEXT_VOCAB_SIZE = data_dict['TEXT'].vocab_size

    # model hyperparameters
    HID_DIM = model_hyperparams_dict['HID_DIM']
    DROPOUT = model_hyperparams_dict['DROPOUT']

    # create model object
    model = MultitaskLogisticRegressionClassificationModel(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=HID_DIM,
        dropout=DROPOUT,
        out_dims=TARGET_SIZES,
    ).to(device)

    return model
