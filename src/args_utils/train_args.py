from argparse import ArgumentParser

from .bert_hparam_args import add_bert_hyperparams_args
from .data_args import add_data_args
from .logistic_regression_hparam_args import add_logistic_regression_hyperparams_args
from .transformer_enc_hparam_args import add_transformer_enc_hyperparams_args


def get_train_args():
    """Cumulative args for cross-lingual infodemic fake news classification model"""
    parser = ArgumentParser(description='infodemic model')
    parser = add_data_args(parser)
    parser = add_transformer_enc_hyperparams_args(parser)
    parser = add_bert_hyperparams_args(parser)
    parser = add_logistic_regression_hyperparams_args(parser)
    parser.add_argument('--log_file_path', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='../models/tmp')
    parser.add_argument('--no_xavier_initialization', action='store_true', default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='transformer_enc')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='whether to freeze the parameters of BERT '
                             '(only applicable when args.model_name is a BERT based model)')
    args = parser.parse_args()

    # sanity check tokenizer
    if args.model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
        assert args.model_name == args.tokenization

    return args
