import os

from data_utils import load_dataframe

if __name__ == "__main__":

    OUT_DIR = os.path.join('../data/prepared')
    os.makedirs(OUT_DIR, exist_ok=True)

    # en
    DATA_DIR_ENGLISH_TRAIN = os.path.join('../data/english/v1/v1')
    DATA_DIR_ENGLISH_DEV = os.path.join('../data/english/v2/v2')
    assert os.path.isdir(DATA_DIR_ENGLISH_TRAIN)
    assert os.path.isdir(DATA_DIR_ENGLISH_DEV)

    for name, data_dir, fname in zip(['train', 'dev'],
                                     [DATA_DIR_ENGLISH_TRAIN, DATA_DIR_ENGLISH_DEV],
                                     ['covid19_disinfo_binary_english_train.tsv',
                                      'covid19_disinfo_binary_english_dev_input.tsv']):
        df = load_dataframe(os.path.join(data_dir, fname))
        # df.rename(columns={'text': 'tweet_text'}, inplace=True)
        df.to_csv(os.path.join(OUT_DIR, f'{name}.en.tsv'), sep='\t', encoding='utf-8', index=False)

    # ar
    DATA_DIR_ARABIC_TRAIN = os.path.join('../data/arabic/v1/v1')
    DATA_DIR_ARABIC_DEV = os.path.join('../data/arabic/v2/v2')
    assert os.path.isdir(DATA_DIR_ARABIC_TRAIN)
    assert os.path.isdir(DATA_DIR_ARABIC_DEV)

    for name, data_dir, fname in zip(['train', 'dev'],
                                     [DATA_DIR_ARABIC_TRAIN, DATA_DIR_ARABIC_DEV],
                                     ['covid19_disinfo_binary_arabic_train.tsv',
                                      'covid19_disinfo_binary_arabic_dev.tsv']):
        df = load_dataframe(os.path.join(data_dir, fname))
        df.rename(columns={'text': 'tweet_text'}, inplace=True)
        df.to_csv(os.path.join(OUT_DIR, f'{name}.ar.tsv'), sep='\t', encoding='utf-8', index=False)

    """
    for name, fname in zip(['train', 'dev'], ['train.tsv', 'dev.tsv']):
        for lang, data_dir in zip(['en', 'ar'], [DATA_DIR_ENGLISH, DATA_DIR_ARABIC]):
            copy(os.path.join(data_dir, fname), os.path.join(OUT_DIR, f'{name}.{lang}.tsv'))
    """

    # bg
    DATA_DIR_BULGARIAN = os.path.join('../data/bulgarian/v1/v1')
    assert os.path.isdir(DATA_DIR_BULGARIAN)

    for name, fname in zip(['train', 'dev'],
                           ['covid19_disinfo_binary_bulgarian_train.tsv', 'covid19_disinfo_binary_bulgarian_dev.tsv']):
        df = load_dataframe(os.path.join(DATA_DIR_BULGARIAN, fname))
        df.rename(columns={'text': 'tweet_text'}, inplace=True)
        df.to_csv(os.path.join(OUT_DIR, f'{name}.bg.tsv'), sep='\t', encoding='utf-8', index=False)

    """
    # ar: change col name from 'text' to 'tweet_text'
    for split_name in ['train', 'dev']:
        df = load_dataframe(os.path.join(DATA_DIR_ARABIC, f'{split_name}.tsv'))
        df.rename(columns={'text': 'tweet_text'}, inplace=True)
        df.to_csv(os.path.join(OUT_DIR, f'{split_name}.ar.tsv'), sep='\t', encoding='utf-8', index=False)
    """
