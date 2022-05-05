"""NOTE: this code requires torch 1.5, transformers 3.0.2"""

import os
import sys
from typing import List

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

sys.path.append('../src')
from data_utils import load_dataframe


def translate(srclang: str, trglang: str, sentences: List[str], beam_size: int) -> List[str]:
    """Translate sentences"""
    model_name = f'Helsinki-NLP/opus-mt-{srclang}-{trglang}'
    print(f'loading tokenizer and model for {model_name}')
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        model = MarianMTModel.from_pretrained(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(89 * '*')
    print(model)
    print('model is on device:', model.device)
    num_params = sum(param.numel() for param in model.parameters())
    print(f'number of parameters: {num_params:,}')
    print(89 * '*')

    all_translations_text = []
    for sent in tqdm(sentences, desc='translate', unit=' sentences'):
        tokens_dict = tokenizer.prepare_translation_batch([sent])
        translations = model.generate(
            input_ids=tokens_dict['input_ids'].to(device),
            attention_mask=tokens_dict['attention_mask'].to(device),
            num_return_sequences=1, num_beams=beam_size, do_sample=False,
        )
        translations_text = [tokenizer.decode(
            t, skip_special_tokens=True
        ) for t in translations]
        all_translations_text.append(translations_text[0])

    return all_translations_text


if __name__ == "__main__":
    BEAM_SIZE = 12
    TRANSLATIONS_DIR = os.path.join('../data/translations')
    os.makedirs(TRANSLATIONS_DIR, exist_ok=True)

    data_paths_dict = {
        'train.en.bg': {'inp': os.path.join('../data/prepared_additional/train.en.tsv'),
                        'out': os.path.join(TRANSLATIONS_DIR, 'train.en.bg.tsv')},
        'train.en.ar': {'inp': os.path.join('../data/prepared_additional/train.en.tsv'),
                        'out': os.path.join(TRANSLATIONS_DIR, 'train.en.ar.tsv')},
        'dev.bg.en': {'inp': os.path.join('../data/prepared_additional/dev.bg.tsv'),
                      'out': os.path.join(TRANSLATIONS_DIR, 'dev.bg.en.tsv')},
        'dev.ar.en': {'inp': os.path.join('../data/prepared_additional/dev.ar.tsv'),
                      'out': os.path.join(TRANSLATIONS_DIR, 'dev.ar.en.tsv')},
        'test.bg.en': {'inp': os.path.join('../data/prepared_test_data/test.bg.tsv'),
                       'out': os.path.join(TRANSLATIONS_DIR, 'test.bg.en.tsv')},
        'test.ar.en': {'inp': os.path.join('../data/prepared_test_data/test.ar.tsv'),
                       'out': os.path.join(TRANSLATIONS_DIR, 'test.ar.en.tsv')},
    }

    for k, v in data_paths_dict.items():
        print(f'*** {k} ***')
        _, src, trg = k.split('.')
        df = load_dataframe(v['inp'])
        trs = translate(srclang=src, trglang=trg, sentences=df['tweet_text'].astype(str).to_list(), beam_size=beam_size)
        df['tweet_text'] = trs
        df.to_csv(v['out'], sep='\t', encoding='utf-8', index=False)
