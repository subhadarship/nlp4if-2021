#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=0
MODELDIR=../models/best_models

# predict
CUDA_VISIBLE_DEVICES=${GPUID} python predict.py --model_dir ${MODELDIR}/en/ --model_name bert-base-uncased --dev_path ../data/prepared_additional/dev.en.tsv --test_inp_path ../data/english/test-input/test-input/covid19_disinfo_binary_english_test_input.tsv --test_pred_path ../predictions/test.en.pred.tsv
CUDA_VISIBLE_DEVICES=${GPUID} python predict.py --model_dir ${MODELDIR}/bg/ --model_name bert-base-multilingual-cased --dev_path ../data/prepared_additional/dev.bg.tsv --test_inp_path ../data/bulgarian/test-input/test-input/covid19_disinfo_binary_bulgarian_test_input.tsv --test_pred_path ../predictions/test.bg.pred.tsv
CUDA_VISIBLE_DEVICES=${GPUID} python predict.py --model_dir ${MODELDIR}/ar --model_name bert-base-multilingual-cased --dev_path ../data/prepared_additional/dev.ar.tsv --test_inp_path ../data/arabic/test-input/test-input/covid19_disinfo_binary_arabic_test_input.tsv --test_pred_path ../predictions/test.ar.pred.tsv

# sanitize predictions
python sanitize_predictions.py ../predictions/test.en.pred.tsv ../predictions/test.san.en.pred.tsv
python sanitize_predictions.py ../predictions/test.bg.pred.tsv ../predictions/test.san.bg.pred.tsv
python sanitize_predictions.py ../predictions/test.ar.pred.tsv ../predictions/test.san.ar.pred.tsv
