import os

import pandas as pd

if __name__ == "__main__":
    write_data_dir = os.path.join('../data/prepared_test_data')
    os.makedirs(write_data_dir, exist_ok=True)

    data_paths_dict = {
        os.path.join(
            '../data/bulgarian/test-gold/test-gold/covid19_disinfo_binary_bulgarian_test_gold.tsv'): os.path.join(
            write_data_dir, 'test.bg.tsv'),
        os.path.join('../data/arabic/test-gold/test-gold/covid19_disinfo_binary_arabic_test_gold.tsv'): os.path.join(
            write_data_dir, 'test.ar.tsv'),
    }

    column_names = ['tweet_no',
                    'tweet_text',
                    'q1_label',
                    'q2_label',
                    'q3_label',
                    'q4_label',
                    'q5_label',
                    'q6_label',
                    'q7_label', ]

    for k, v in data_paths_dict.items():
        df = pd.read_csv(k, sep='\t', encoding='utf-8', na_filter=False, names=column_names)
        df.to_csv(v, sep='\t', encoding='utf-8', index=False)
