#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1

# Multilingual BERT

MODELNAME=bert-base-multilingual-cased

MODELDIR=/mnt/backup/panda/nlp4if-2021/models


# BERT parameters are trainable

for TRGLANG in bg ar; do
  for HIDDIM in 128 256 512; do
    for LR in 0.0005 0.005 0.05; do
      # zero
      expt=en${TRGLANG}/${MODELNAME}/zero_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # few 50
      expt=en${TRGLANG}/${MODELNAME}/few50_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_50 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # few 100
      expt=en${TRGLANG}/${MODELNAME}/few100_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_100 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # few 150
      expt=en${TRGLANG}/${MODELNAME}/few150_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_150 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # few 200
      expt=en${TRGLANG}/${MODELNAME}/few200_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_200 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # full
      expt=en${TRGLANG}/${MODELNAME}/full_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

      # trg
      expt=en${TRGLANG}/${MODELNAME}/trg_fc${HIDDIM}_lr${LR}_trainable
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples ${TRGLANG}_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10
      rm -rf ${MODELDIR}/${expt}

    done

  done
done

# BERT parameters are frozen

for TRGLANG in bg ar; do
  for HIDDIM in 128 256 512; do
    for LR in 0.0005 0.005 0.05; do
      # zero
      expt=en${TRGLANG}/${MODELNAME}/zero_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # few 50
      expt=en${TRGLANG}/${MODELNAME}/few50_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_50 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # few 100
      expt=en${TRGLANG}/${MODELNAME}/few100_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_100 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # few 150
      expt=en${TRGLANG}/${MODELNAME}/few150_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_150 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # few 200
      expt=en${TRGLANG}/${MODELNAME}/few200_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_200 --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # full
      expt=en${TRGLANG}/${MODELNAME}/full_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

      # trg
      expt=en${TRGLANG}/${MODELNAME}/trg_fc${HIDDIM}_lr${LR}_frozen
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples ${TRGLANG}_all --trglang ${TRGLANG} --tokenization ${MODELNAME} --model_name ${MODELNAME} --bert_fc_dim ${HIDDIM} --model_dir ${MODELDIR}/${expt} --log_file_path ../logs/${expt}.txt --lr ${LR} --batch_size 1024 --max_epochs 999 --early_stopping_patience 10 --freeze_bert
      rm -rf ${MODELDIR}/${expt}

    done

  done
done
