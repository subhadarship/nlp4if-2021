import os

from data_utils import load_dataframe

if __name__ == "__main__":
    DATA_DIR = '../data/prepared'
    REFERENCES_DIR = '../references'

    # dev data
    for lang in ['en', 'bg', 'ar']:
        df = load_dataframe(os.path.join(DATA_DIR, f'dev.{lang}.tsv'))
        df = df.filter([f'q{idx + 1}_label' for idx in range(7)])
        df.to_csv(
            os.path.join(REFERENCES_DIR, f'dev.{lang}.ref.tsv'),
            sep='\t', encoding='utf-8', index=False, header=False
        )
