#!/bin/bash

cd ../src
echo "WORKING DIR: $PWD"

GPUID=1

# logistic regression

for TRGLANG in bg ar; do
  for HIDDIM in 128 256 512; do
    for VOCAB in 32000 16000 8000; do
      # zero
      expt=${TRGLANG}/zero_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # few 50
      expt=${TRGLANG}/few50_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_50 --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # few 100
      expt=${TRGLANG}/few100_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_100 --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # few 150
      expt=${TRGLANG}/few150_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_150 --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # few 200
      expt=${TRGLANG}/few200_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_200 --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # full
      expt=${TRGLANG}/full_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples en_all,${TRGLANG}_all --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

      # trg
      expt=${TRGLANG}/trg_hidden${HIDDIM}_vocab${VOCAB}
      CUDA_VISIBLE_DEVICES=${GPUID} python train.py --srclangs_with_num_samples ${TRGLANG}_all --trglang ${TRGLANG} --tokenization tweet --model_name logistic_regression --logistic_regression_hid_dim ${HIDDIM} --model_dir ../models/${expt} --log_file_path ../logs/${expt} --max_vocab_size ${VOCAB} --batch_size 4096 --max_epochs 999 --early_stopping_patience 10

    done

  done
done
