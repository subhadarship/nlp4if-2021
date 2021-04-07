import os
import sys

import pandas as pd


def sanitize(inp_fpath: str, out_fpath: str) -> None:
    """Sanitize predictions in input file and write to output file"""
    assert os.path.isfile(inp_fpath)
    df = pd.read_csv(inp_fpath, sep='\t', encoding='utf-8', na_filter=False, header=False)
    for col_idx in [0, 5, 6]:
        col = df[col_idx].to_list()
        sanitized_col = []
        num_changed = 0
        for label in col:
            assert label in ['yes', 'no', 'nan']
            if label == 'nan':
                sanitized_col.append('no')
                num_changed += 1
            else:
                sanitized_col.append(label)
        print(f'{num_changed} out of {len(col)} labels were changed to "no" for question {col_idx + 1}')
        df[col_idx] = sanitized_col

    # write to file
    df.to_csv(out_fpath, sep='\t', encoding='utf-8', index=False, header=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise AssertionError(f'Usage: python sanitize_predictions.py inp.tsv out.tsv')
    sanitize(sys.argv[1], sys.argv[2])
