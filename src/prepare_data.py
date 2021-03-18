import os
from shutil import copy

if __name__ == "__main__":

    OUT_DIR = os.path.join('../data/prepared')
    os.makedirs(OUT_DIR, exist_ok=True)

    # en and ar
    DATA_DIR_ENGLISH = os.path.join('../data/data_english_v1/v1')
    DATA_DIR_ARABIC = os.path.join('../data/data_arabic_v1/v1')
    assert os.path.isdir(DATA_DIR_ENGLISH)
    assert os.path.isdir(DATA_DIR_ARABIC)

    for name, fname in zip(['train', 'dev'], ['train.tsv', 'dev.tsv']):
        for lang, data_dir in zip(['en', 'ar'], [DATA_DIR_ENGLISH, DATA_DIR_ARABIC]):
            copy(os.path.join(data_dir, fname), os.path.join(OUT_DIR, f'{name}.{lang}.tsv'))

    # bg
    DATA_DIR_BULGARIAN = os.path.join('../data/data_bulgarian_v1/v1')
    assert os.path.isdir(DATA_DIR_BULGARIAN)

    for name, fname in zip(['train', 'dev'],
                           ['covid19_disinfo_binary_bulgarian_train.tsv', 'covid19_disinfo_binary_bulgarian_dev.tsv']):
        copy(os.path.join(DATA_DIR_BULGARIAN, fname), os.path.join(OUT_DIR, f'{name}.bg.tsv'))
