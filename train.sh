#!/bin/bash

devices=$1
save_name=$2

CUDA_VISIBLE_DEVICES=$devices python -u train.py --dataset ./data \
--glove_embed_path ./data/glove.42B.300d.txt \
--cuda \
--epoch 50 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 1 \
--seed 900 \
--save ${save_name} \
--embed_size 600 \
--col_embed_size 600 \
--sentence_features \
--hidden_size 600 \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size 600 \
--lr 0.0001 \
--bert 0 \
--batch_size 6 \
#--fasttext_together \
#--fasttext wiki
