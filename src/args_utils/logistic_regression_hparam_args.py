def add_logistic_regression_hyperparams_args(parser):
    """Only applicable when args.model_name is 'logistic_regression'"""
    parser.add_argument('--logistic_regression_hid_dim', type=int, default=128)
    parser.add_argument('--logistic_regression_dropout', type=float, default=0.1)
    return parser
