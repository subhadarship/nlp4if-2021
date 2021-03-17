import logging
from typing import List

from data_utils import LabelField

logger = logging.getLogger(__name__)


def postprocess_labels(labels: List[List[int]], label_fields: List[LabelField]) -> List[List[str]]:
    """Postprocess labels. First convert ints to corresponding strings.
    Then apply sanity postprocessing for q2, q3, q4, q5"""
    out = []
    for li in labels:
        out.append([label_fields[idx].itos[li[idx]] for idx in range(7)])

    out = sanity_postprocess(out)
    return out


def sanity_postprocess(inp_list: List[List[str]]) -> List[List[str]]:
    """Change q2, q3, q4, q5 predictions based on q1's prediction"""
    postprocessed = []
    num_total, num_changed = 0, 0
    for row in inp_list:
        post = []
        q1_pred = row[0]
        for col_idx, item in enumerate(row):
            if col_idx in [1, 2, 3, 4]:
                num_total += 1
                if q1_pred == 'no':
                    post.append('nan')
                    if item != 'nan':
                        num_changed += 1
                else:
                    post.append(item)
            else:
                post.append(item)
        postprocessed.append(post)

    logger.info(
        f'{num_changed} ({100 * num_changed / num_total:0.2f} %) out of {num_total} q2, q3, q4, q5 predictions are changed during postprocessing')
    return postprocessed
