from argparse import ArgumentParser


def get_predict_args():
    """Prediction args for cross-lingual infodemic fake news classification model"""
    parser = ArgumentParser(description='prediction using infodemic model')
    parser.add_argument('--log_file_path', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='../models/tmp')
    parser.add_argument('--model_name', type=str, default='transformer_enc')
    parser.add_argument('--dev_path', type=str, default=None)
    parser.add_argument('--test_inp_path', type=str, default=None)
    parser.add_argument('--test_pred_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4096)
    args = parser.parse_args()

    return args
