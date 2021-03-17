def add_bert_hyperparams_args(parser):
    """Only applicable when args.model_name is a BERT based model"""
    parser.add_argument('--bert_fc_dim', type=int, default=64, help='hidden size of the linear layer added on top')

    return parser
