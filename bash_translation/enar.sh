#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1

BERTMODELNAME=bert-base-uncased
MBERTMODELNAME=bert-base-multilingual-cased

MODELDIR=../models/translation
LOGDIR=../logs_translation

# Pre-trained parameters are frozen

for TRGLANG in ar; do
  for HIDDIM in 128 256 512; do
    for LR in 0.0005 0.005 0.05; do

      # baseline BERT
      expt=en${TRGLANG}/zero/baseline_bert/fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py \
        --train_data_dir ../data/prepared_additional \
        --dev_data_dir ../data/prepared_additional \
        --test_data_dir ../data/prepared_test_data \
        --srclangs_with_num_samples en_all --trglang ${TRGLANG} --tokenization ${BERTMODELNAME} --model_name ${BERTMODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # baseline mBERT
      expt=en${TRGLANG}/zero/baseline_mbert/fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py \
        --train_data_dir ../data/prepared_additional \
        --dev_data_dir ../data/prepared_additional \
        --test_data_dir ../data/prepared_test_data \
        --srclangs_with_num_samples en_all --trglang ${TRGLANG} --tokenization ${MBERTMODELNAME} --model_name ${MBERTMODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # translate-test BERT
      expt=en${TRGLANG}/zero/translate_test_bert/fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py \
        --train_data_dir ../data/prepared_additional \
        --dev_data_dir ../data/translations \
        --test_data_dir ../data/translations \
        --srclangs_with_num_samples en_all --trglang ${TRGLANG}.en --tokenization ${BERTMODELNAME} --model_name ${BERTMODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # translate-test mBERT
      expt=en${TRGLANG}/zero/translate_test_mbert/fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py \
        --train_data_dir ../data/prepared_additional \
        --dev_data_dir ../data/translations \
        --test_data_dir ../data/translations \
        --srclangs_with_num_samples en_all --trglang ${TRGLANG}.en --tokenization ${MBERTMODELNAME} --model_name ${MBERTMODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # translate-train mBERT
      expt=en${TRGLANG}/zero/translate_train_mbert/fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py \
        --train_data_dir ../data/translations \
        --dev_data_dir ../data/prepared_additional \
        --test_data_dir ../data/prepared_additional \
        --srclangs_with_num_samples en.${TRGLANG}_all --trglang ${TRGLANG} --tokenization ${MBERTMODELNAME} --model_name ${MBERTMODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ${LOGDIR}/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

    done
  done
done
