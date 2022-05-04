import os

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List
from tqdm import tqdm



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
            num_return_sequences=1, num_beams=12, do_sample=False,
        )
        translations_text = [tokenizer.decode(
            t, skip_special_tokens=True
        ) for t in translations]
        all_translations_text.append(translations_text[0])

    return all_translations_text

if __name__ == "__main__":
    translations_dir = os.path.join('../data/translation_prepared_additional')
    os.makedirs(translations_dir, exist_ok=True)

