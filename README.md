# Cross-lingual misinformation detection

This repo contains the code for cross-lingual misinformation detection. See paper
in [`NLP4IF_2021.pdf`](https://github.com/subhadarship/nlp4if-2021/blob/main/NLP4IF_2021.pdf).

## Quick start

Install PyTorch 1.1.0 from the [official website](https://pytorch.org/). Install other dependencies
in `requirements.txt`.

### Prepare data

For details of the data, see

- https://gitlab.com/NLP4IF/nlp4if-2021
- https://www.aclweb.org/portal/content/nlp4if-2021-shared-tasks

```
cd src
python prepare_data.py  # prepare data without using additional data
python prepare_data_additional.py  # prepare data without using additional data
```

Analysis of the data is available in `notebooks/analyze_data.ipynb` and `notebooks/analyze_data_additional.ipynb`.

### Training

Choose the appropriate file in the `bash` folder to train without using additional data or the folder `bash_additional`
to use additional data for training. For example, if you want to fine-tune multilingual BERT with source language
English while using the additional data, run the following command lines.

```
cd bash_additional
chmod +x train_multilingual_bert_src_en.sh
./train_multilingual_bert_src_en.sh
```

The training logs are saved in the specified file, the argument for which is `--log_file_path`. The log file also stores
the evaluation results after training completes.

**Note**: To tabulate the results from the log files and pick the best hyperparameters across multiple runs,
see `notebooks/tabulate_results_v{1,2,3}.ipynb`.

### Predict labels for the test set

```
cd bash_predict
chmod +x predict_best_sys.sh
./predict_best_sys.sh
```

#### Training logs

- `logs_v1` contains the training logs while using own train-dev splits for en and ar and provided train and dev data
  for bg.
- `logs_v2` contains the training logs while using the provided train and dev data for all languages.
- `logs` contains the training logs while using the provided additional train and dev data for all languages.

## Citation

```
@inproceedings{detecting-multilingual-misinformation,
    title = "Detecting Multilingual {COVID}-19 Misinformation on Social Media via Contextualized Embeddings",
    author = "Panda, Subhadarshi and Levitan, Sarah Ita",
    booktitle = "Proceedings of the Fourth Workshop on Natural Language Processing for Internet Freedom: Censorship, Disinformation, and Propaganda",
    series = {NLP4IF@NAACL'~21},
    month = {June},
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
