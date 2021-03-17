import logging
import os
import time

import torch
from torch import nn

from args_utils import get_train_args
from data_utils import (
    load_dataframe,
    preprocess_bert_data,
    preprocess_data,
)
from model_utils import (
    count_parameters,
    load_transformer_enc_multitask_classification_model,
    load_bert_multitask_classification_model,
    load_logistic_regression_multitask_classification_model,
    save_checkpoint,
    load_checkpoint,
    initialize_weights,
)
from training_utils import (
    init_logger,
    seed_everything,
    epoch_time,
    train,
    evaluate,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # get train args
    args = get_train_args()

    # init logger
    init_logger(args.log_file_path)
    logger.info("\n\n*****************\n***RUN STARTED***\n*****************\n")

    # log args
    args_str = f'args\n{89 * "-"}\n'
    for k, v in args.__dict__.items():
        args_str += f'\t{k}: {v}\n'
    args_str += f'{89 * "-"}\n'
    logger.info(args_str)

    # set random seed
    seed_everything(args.random_seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    """Load data"""

    train_df = load_dataframe(fpath=os.path.join(args.train_data_dir, 'train.tsv'))
    val_df = load_dataframe(fpath=os.path.join(args.train_data_dir, 'dev.tsv'))
    test_df = None

    """Preprocess data"""

    if args.model_name == 'transformer_enc':
        data_dict = preprocess_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tokenization_name=args.tokenization,
            sos_token='<sos>',
            max_len=1000,
            batch_size=args.batch_size,
            max_vocab_size=None,
            device=device,
        )
    elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
        data_dict = preprocess_bert_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tokenization_name=args.tokenization,
            batch_size=args.batch_size,
            device=device,
        )
    elif args.model_name == 'logistic_regression':
        data_dict = preprocess_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tokenization_name=args.tokenization,
            sos_token=None,
            max_len=None,
            batch_size=args.batch_size,
            max_vocab_size=None,
            device=device,
        )
    else:
        raise NotImplementedError

    """Load model"""

    if args.model_name == 'transformer_enc':
        model_hyperparams_dict = {
            'HID_DIM': args.hid_dim,
            'ENC_LAYERS': args.num_enc_layers,
            'ENC_HEADS': args.num_enc_heads,
            'ENC_PF_DIM': args.enc_pf_dim,
            'ENC_DROPOUT': args.enc_dropout,
            'FC_DIM': args.fc_dim,
        }
        model = load_transformer_enc_multitask_classification_model(
            model_hyperparams_dict=model_hyperparams_dict,
            data_dict=data_dict,
            device=device,
        )
    elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
        model_hyperparams_dict = {
            'FC_DIM': args.bert_fc_dim,
        }
        model = load_bert_multitask_classification_model(
            model_name=args.model_name,
            model_hyperparams_dict=model_hyperparams_dict,
            data_dict=data_dict,
            freeze=args.freeze_bert,
            device=device,
        )
    elif args.model_name == 'logistic_regression':
        model_hyperparams_dict = {
            'HID_DIM': args.logistic_regression_hid_dim,
            'DROPOUT': args.logistic_regression_dropout,
        }
        model = load_logistic_regression_multitask_classification_model(
            model_hyperparams_dict=model_hyperparams_dict,
            data_dict=data_dict,
            device=device,
        )
    else:
        raise NotImplementedError

    # log model
    logger.info(f'model\n{89 * "-"}\n{str(model)}\n{89 * "-"}\n')
    logger.info(f'the model has {count_parameters(model):,} trainable parameters')

    """Train"""

    if not args.no_xavier_initialization:
        if args.model_name in ['transformer_enc', 'logistic_regression']:
            logger.info(f'applying xavier initialization of model parameters')
            model.apply(initialize_weights)
        elif args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
            pass
        else:
            raise NotImplementedError

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    valid_losses = []

    logger.info('🌋  starting training..')

    for epoch in range(1, args.max_epochs + 1):

        start_time = time.time()

        train_loss = train(
            model=model,
            iterator=data_dict['train_iter'],
            optimizer=optimizer,
            criterion=criterion,
            clip=args.clip,
            curr_epoch=epoch,
            max_epochs=args.max_epochs,
        )
        valid_loss, valid_metrics = evaluate(
            model=model,
            iterator=data_dict['val_iter'],
            criterion=criterion,
        )
        valid_losses.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(
            f'Epoch: {epoch:04} | Time: {epoch_mins}m_{epoch_secs}s | '
            f'train_loss: {train_loss:.3f} | '
            f'val_loss: {valid_loss:.3f}'
        )
        logger.info(f'📣 val metrics 📣\n{create_table(valid_metrics)}')

        if valid_loss <= best_valid_loss:
            logger.info('\t--Found new best val loss')
            best_valid_loss = valid_loss
            save_checkpoint(
                checkpoint_dict={
                    'data_dict': {
                        key: val for key, val in data_dict.items() if key.isupper()  # only save fields
                    },
                    'model_hyperparams_dict': model_hyperparams_dict,
                    'model_state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                model_dir=args.model_dir,
            )

        # early stopping when the best val loss is lower than
        # the last args.early_stopping_patience val losses
        if args.early_stopping_patience is not None and \
                len(valid_losses) > args.early_stopping_patience and \
                all(best_valid_loss < loss for loss in valid_losses[-args.early_stopping_patience:]):
            logger.info('\t--STOPPING EARLY')
            break

    """Evaluate"""

    # load checkpoint
    logger.info(f'load checkpoint from {args.model_dir}')
    checkpoint_dict = load_checkpoint(args.model_dir, device)

    # load model weights
    logger.info(f'load model weights from checkpoint in {args.model_dir}')
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    # compute val loss
    best_valid_loss, best_valid_metrics = evaluate(args, model, data_dict['val_iter'], criterion)
    logger.info(
        f'best_val_loss: {best_valid_loss:.3f}'
    )
    logger.info(f'📣 best validation metrics 📣\n{create_table(best_valid_metrics)}')

    # compute test loss
    logger.info(f'🔥 start testing..')
    test_loss, test_metrics = evaluate(args, model, data_dict['test_iter'], criterion)
    logger.info(
        f'test_loss: {test_loss:.3f}'
    )
    logger.info(f'📣 test metrics 📣\n{create_table(test_metrics)}')
