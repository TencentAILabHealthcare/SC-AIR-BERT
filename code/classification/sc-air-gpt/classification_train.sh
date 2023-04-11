#!/usr/bin/env bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
export PYTHONPATH=$SHELL_FOLDER
echo "current path " $SHELL_FOLDER

export CUDA_VISIBLE_DEVICES='0'

python3 classification.py \
-c ./data/classification/GPT_10x/ELAGIGILTV/train.tsv \
-d ./data/classification/GPT_10x/ELAGIGILTV/valid.tsv \
-t ./data/classification/GPT_10x/ELAGIGILTV/test.tsv \
--block_size 77 \
--train_label ./data/classification/GPT_10x/ELAGIGILTV/train_label.tsv \
--valid_label ./data/classification/GPT_10x/ELAGIGILTV/valid_label.tsv \
--test_label ./data/classification/GPT_10x/ELAGIGILTV/test_label.tsv \
--class_name ELAGIGILTV \
--vocab_file ./data/pretrain/3-mer_sep_pad/gpt_tokenizer-vocab.json \
--merges_file ./data/pretrain/3-mer_sep_pad/gpt_tokenizer-merges.txt \
--gpt_model ./checkpoint/sc-air-gpt/pretrain_models/checkpoint-92250 \
--in_features 512 \
-o ./result/classification/10x/ELAGIGILTV \
--lr_b 0.0001 \
--lr_c 0.001 \
--batch_size 32 \
--finetune 1 \
--seed 63