from typing import List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(labels: List[int], predictions: List[int]) -> dict:
    """Compute metrics"""
    metrics_dict = {
        'acc': accuracy_score(y_true=labels, y_pred=predictions),
        'f1': f1_score(y_true=labels, y_pred=predictions, average='weighted'),
        'precision': precision_score(y_true=labels, y_pred=predictions, average='weighted'),
        'recall': recall_score(y_true=labels, y_pred=predictions, average='weighted'),
    }
    return metrics_dict
