import logging
from typing import Union, Tuple, Dict

import torch
from tqdm import tqdm

from data_utils import SMARTTOKDataLoader
from model_utils import (
    MultitaskTransformerEncoderClassificationModel,
    MultitaskBertClassificationModel,
    MultitaskLogisticRegressionClassificationModel,
)
from .eval_metrics import compute_metrics

logger = logging.getLogger(__name__)


def train(
        model: Union[
            MultitaskTransformerEncoderClassificationModel, MultitaskBertClassificationModel, MultitaskLogisticRegressionClassificationModel],
        iterator: Union[SMARTTOKDataLoader],
        optimizer, criterion, clip: float, curr_epoch: int, max_epochs: int
) -> float:
    """Train method"""

    # set model to train mode
    model.train()

    epoch_loss = 0.0

    tqdm_meter = tqdm(
        iterator,
        unit=' batches',
        desc=f'[EPOCH {curr_epoch}/{max_epochs}]',
        leave=False,
        total=0,
    )

    for batch_idx, batch in enumerate(tqdm_meter):
        tqdm_meter.total = len(iterator)
        text = batch['text']
        targets = batch['labels']

        # zero out optimizer
        optimizer.zero_grad()

        # forward pass
        outs = model(text)

        logits = outs['logits']

        # logits = {q1: [batch size, output dim], ...}
        # targets = {q1: [batch size], ...}

        # compute loss
        losses = [criterion(logits[f'q{idx + 1}'], targets[f'q{idx + 1}']) for idx in range(7)]

        loss = sum(losses)

        # backward
        loss.backward()

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # optimizer step
        optimizer.step()

        epoch_loss += loss.item()

        # update tqdm meter
        tqdm_meter.set_postfix(
            ordered_dict={
                'loss': f'{loss.item():0.4f}',
            }
        )
        tqdm_meter.update()

    return epoch_loss / (batch_idx + 1)


def evaluate(
        model: Union[
            MultitaskTransformerEncoderClassificationModel, MultitaskBertClassificationModel, MultitaskLogisticRegressionClassificationModel],
        iterator: Union[SMARTTOKDataLoader], criterion
) -> Tuple[float, Dict[str, float]]:
    """Evaluate method"""

    # set model to eval mode
    model.eval()

    epoch_loss = 0.0

    preda, predb, predc = [], [], []
    labela, labelb, labelc = [], [], []

    val_meter = tqdm(iterator, desc='eval', unit=' batches', leave=False, total=0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_meter):
            val_meter.total = len(iterator)
            text = batch['text']
            targets = batch['labels']

            # forward pass
            outs = model(text)

            logits = outs['logits']

            # logits = {q1: [batch size, output dim], ...}
            # targets = {q1: [batch size], ...}

            # compute loss
            losses = [criterion(logits[f'q{idx + 1}'], targets[f'q{idx + 1}']) for idx in range(7)]

            loss = sum(losses)

            epoch_loss += loss.item()

            # compute prediction
            pred_tuples = [logits[f'q{idx + 1}'].max(dim=1) for idx in range(7)]
            preds = [pred for _, pred in pred_tuples]

            preda.extend(preda_batch.detach().cpu().tolist())
            labela.extend(targeta.detach().cpu().tolist())
            predb.extend(predb_batch.detach().cpu().tolist())
            labelb.extend(targetb.detach().cpu().tolist())
            predc.extend(predc_batch.detach().cpu().tolist())
            labelc.extend(targetc.detach().cpu().tolist())

    # compute metric
    metrics = {
        'two-way': compute_metrics(labela, preda),
        'three-way': compute_metrics(labelb, predb),
        'six-way': compute_metrics(labelc, predc),
    }
    return epoch_loss / (batch_idx + 1), metrics
