#!/bin/bash

python train.py \
    --model_name_or_path leo4life/algoml-bert-tiny \
    --tokenizer_name leo4life/algoml-bert-tokenizer \
    --dataset_name leo4life/algoml_bookcorpus_10p \
    --validation_split_percentage 1 \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --output_dir ./bertlsh

if [ $? -eq 0 ]; then
    python train.py \
        --model_name_or_path leo4life/algoml-bert-tiny \
        --tokenizer_name leo4life/algoml-bert-tokenizer \
        --dataset_name leo4life/algoml_bookcorpus_10p \
        --validation_split_percentage 1 \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --output_dir ./origbert
else
    echo "First training command failed. Exiting."
    exit 1
fi