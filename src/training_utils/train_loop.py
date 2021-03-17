import logging
from typing import Union, Tuple, Dict, List

import torch
from tqdm import tqdm

from data_utils import LabelField
from data_utils import SMARTTOKDataLoader
from model_utils import (
    MultitaskTransformerEncoderClassificationModel,
    MultitaskBertClassificationModel,
    MultitaskLogisticRegressionClassificationModel,
)
from .eval_metrics import compute_metrics
from .postprocess import postprocess_labels

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
        iterator: Union[SMARTTOKDataLoader], criterion,
        label_fields: List[LabelField],
        all_classes: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Evaluate method"""

    # set model to eval mode
    model.eval()

    epoch_loss = 0.0

    all_targets, all_preds = [], []

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
            preds = {k: v.max(dim=1)[1] for k, v in logits.items()}

            all_targets.append({k: v.detach().cpu() for k, v in targets.items()})
            all_preds.append({k: v.detach().cpu() for k, v in preds.items()})

    # flatten
    flattened_targets = torch.cat(
        [torch.cat([_t[f'q{idx + 1}'].view(-1, 1) for idx in range(7)], dim=1) for _t in all_targets]).tolist()
    flattened_preds = torch.cat(
        [torch.cat([_p[f'q{idx + 1}'].view(-1, 1) for idx in range(7)], dim=1) for _p in all_preds]).tolist()

    # postprocess
    logger.info('postprocessing targets..')
    flattened_targets_postprocessed = postprocess_labels(flattened_targets, label_fields)
    logger.info('postprocessing predictions..')
    flattened_preds_postprocessed = postprocess_labels(flattened_preds, label_fields)
    # compute metric
    metrics = compute_metrics(
        gold_labels=flattened_targets_postprocessed,
        predictions=flattened_preds_postprocessed,
        all_classes=all_classes
    )
    return epoch_loss / (batch_idx + 1), metrics
