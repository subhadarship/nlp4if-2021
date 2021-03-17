def add_transformer_enc_hyperparams_args(parser):
    """Only applicable when args.model_name is 'transformer_enc'"""
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--num_enc_layers', type=int, default=3)
    parser.add_argument('--num_enc_heads', type=int, default=8)
    parser.add_argument('--enc_pf_dim', type=int, default=256)
    parser.add_argument('--enc_dropout', type=float, default=0.1)
    parser.add_argument('--fc_dim', type=int, default=64, help='hidden size of the linear layer added on top')

    return parser
