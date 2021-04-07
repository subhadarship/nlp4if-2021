import logging
import os
from typing import Union, List

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from args_utils import get_predict_args
from data_utils import (
    load_dataframe,
    InfodemicDataset,
    BertInfodemicDataset,
    SMARTTOKDataLoader,
    LabelField,
)
from model_utils import (
    load_transformer_enc_multitask_classification_model,
    load_bert_multitask_classification_model,
    load_logistic_regression_multitask_classification_model,
    load_checkpoint,
    MultitaskBertClassificationModel,
    MultitaskLogisticRegressionClassificationModel,
    MultitaskTransformerEncoderClassificationModel,
)
from training_utils import (
    init_logger,
    evaluate,
)
from training_utils import postprocess_labels

logger = logging.getLogger(__name__)


def predict(
        _model: Union[
            MultitaskTransformerEncoderClassificationModel,
            MultitaskBertClassificationModel,
            MultitaskLogisticRegressionClassificationModel
        ],
        iterator: Union[SMARTTOKDataLoader],
        label_fields: List[LabelField],
) -> List[List[str]]:
    """Predict method"""

    # set model to eval mode
    _model.eval()

    all_preds = []

    test_meter = tqdm(iterator, desc='predict', unit=' batches', leave=False, total=0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_meter):
            test_meter.total = len(iterator)
            text = batch['text']

            # forward pass
            outs = _model(text)

            logits = outs['logits']

            # logits = {q1: [batch size, output dim], ...}

            # compute prediction
            preds = {k: v.max(dim=1)[1] for k, v in logits.items()}

            all_preds.append({k: v.detach().cpu() for k, v in preds.items()})

    # flatten
    flattened_preds = torch.cat(
        [torch.cat([_p[f'q{idx + 1}'].view(-1, 1) for idx in range(7)], dim=1) for _p in all_preds]).tolist()

    # postprocess
    logger.info('postprocessing predictions..')
    flattened_preds_postprocessed = postprocess_labels(flattened_preds, label_fields)

    return flattened_preds_postprocessed


if __name__ == "__main__":

    # get predict args
    args = get_predict_args()

    # init logger
    init_logger(args.log_file_path)
    logger.info("\n\n*****************\n***RUN STARTED***\n*****************\n")

    # log args
    args_str = f'args\n{89 * "-"}\n'
    for k, v in args.__dict__.items():
        args_str += f'\t{k}: {v}\n'
    args_str += f'{89 * "-"}\n'
    logger.info(args_str)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    """Load data"""
    data_dfs_dict = {
        'dev': load_dataframe(args.dev_path) if args.dev_path is not None else None,
        'test': load_dataframe(args.test_inp_path) if args.test_inp_path is not None else None,
    }

    if data_dfs_dict['test'] is not None:
        # rename column
        data_dfs_dict['test'].rename(columns={'text': 'tweet_text'}, inplace=True)
        # add dummy labels
        for idx in range(7):
            data_dfs_dict[f'q{idx + 1}_label'] = ['yes'] * len(data_dfs_dict['test'])

    """Load checkpoint"""
    # load checkpoint
    logger.info(f'load checkpoint from {args.model_dir}')
    checkpoint_dict = load_checkpoint(args.model_dir, device)
    TEXT = checkpoint_dict['data_dict']['TEXT']
    LABELS = checkpoint_dict['data_dict']['LABELS']

    """Preprocess data"""
    datasets_dict, dataloaders_dict = {}, {}
    for split_name in data_dfs_dict:
        if data_dfs_dict[split_name] is None:
            continue
        if args.model_name == 'transformer_enc':
            datasets_dict[split_name] = InfodemicDataset(
                df=data_dfs_dict[split_name],
                text_field=TEXT,
                label_fields=LABELS,
                build_vocab=False,
                max_len=1000,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=TEXT.stoi[TEXT.pad_token],
                shuffle=False,
                progress_bar=True,
                device=device
            )
        elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
            datasets_dict[split_name] = BertInfodemicDataset(
                df=data_dfs_dict[split_name],
                bert_tokenizer=checkpoint_dict['data_dict']['TOKENIZER'],
                label_fields=LABELS,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=checkpoint_dict['data_dict']['TOKENIZER'].pad_token_id,
                shuffle=False,
                progress_bar=True,
                device=device
            )
        elif args.model_name == 'logistic_regression':
            datasets_dict[split_name] = InfodemicDataset(
                df=data_dfs_dict[split_name],
                text_field=TEXT,
                label_fields=LABELS,
                build_vocab=False,
                max_len=None,
            )
            dataloaders_dict[split_name] = SMARTTOKDataLoader(
                dataset=datasets_dict[split_name],
                max_tokens=args.batch_size,
                pad_idx=TEXT.stoi[TEXT.pad_token],
                shuffle=False,
                progress_bar=True,
                device=device
            )
        else:
            raise NotImplementedError

    """Load model"""

    if args.model_name == 'transformer_enc':
        model = load_transformer_enc_multitask_classification_model(
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            device=device,
        )
    elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
        model = load_bert_multitask_classification_model(
            model_name=args.model_name,
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            freeze=True,  # setting to True/False does not affect predictions
            device=device,
        )
    elif args.model_name == 'logistic_regression':
        model = load_logistic_regression_multitask_classification_model(
            model_hyperparams_dict=checkpoint_dict['model_hyperparams_dict'],
            data_dict=checkpoint_dict['data_dict'],
            device=device,
        )
    else:
        raise NotImplementedError

    # log model
    logger.info(f'model\n{89 * "-"}\n{str(model)}\n{89 * "-"}\n')
    logger.info(
        f'the model has '
        f'{sum(p.numel() for p in model.parameters()):,} '
        f'total parameters (both trainable/non-trainable)'
    )

    """Predict"""
    # criterion
    criterion = nn.CrossEntropyLoss()

    # load model weights
    logger.info(f'load model weights from checkpoint in {args.model_dir}')
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    if 'dev' in dataloaders_dict:
        # compute val loss
        logger.info(f'🔥 start prediction on dev inputs..')
        valid_loss, valid_metrics = evaluate(model=model, iterator=dataloaders_dict['dev'], criterion=criterion,
                                             label_fields=LABELS, all_classes=["yes", "no"], )
        logger.info(f'val_loss: {valid_loss:.3f}')
        logger.info(f'📣 validation metrics 📣 {valid_metrics}')

    if 'test' in dataloaders_dict:
        # predict on test inputs
        logger.info(f'🔥 start prediction on test inputs..')
        test_predictions = predict(_model=model, iterator=dataloaders_dict['test'], label_fields=LABELS)
        if args.test_pred_path is not None:
            if os.path.basename(args.test_pred_path) != '':
                os.makedirs(os.path.basename(args.test_pred_path), exist_ok=True)
            logger.info(f'writing test predictions to {args.test_pred_path}..')
            test_predictions_df = pd.DataFrame(test_predictions)
            test_predictions_df.to_csv(
                args.test_predictions_df,
                sep='\t', encoding='utf-8', index=False, header=False
            )
