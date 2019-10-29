#!/bin/bash

devices=$1
save_name=$2

CUDA_VISIBLE_DEVICES=$devices python -u train.py --dataset ./data \
--glove_embed_path ./data/glove.42B.300d.txt \
--cuda \
--epoch 1000 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 1 \
--seed 4 \
--save ${save_name} \
--embed_size 300 \
--sentence_features \
--column_pointer \
--hidden_size 300 \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size 300 \
--load_model saved_model/progressive1571893963/\{202\}_\{0.49295863818199814\}.model