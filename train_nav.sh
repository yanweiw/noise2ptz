#!/bin/bash

## train on 1k (Superior) and valid on 1k (Crandon) # concatdataset samples 20 traj at max per batch so bsize 32 is sufficient
# python train_nav.py --run 2k_ptz_ff_1 --base_dir data/nav_train --train_env Superior --valid_env Crandon \
		# --train_dir nav0 --valid_dir nav0 --PTZ_weights weights/noise.pth --lr 0.001 --bsize 32 --seq_len 1

# # train on 50k (10 env) and valid on 10k (10 env) # one batch has 1000 datapoints
# python train_nav.py --run 60k_lstm_15 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 \
# 					--lr 0.001 --bsize 128 --seq_len 15 --with_lstm --data_parallel

# train on 1k (Superior) and valid on 1k (Crandon) with lstm
# python train_nav.py --run 2k_lstm_15 --base_dir data/nav_train --train_env Superior --valid_env Crandon \
					# --train_dir nav0 --valid_dir nav0 --lr 0.001 --bsize 32 --seq_len 15 --with_lstm 

# train on 1k (Superior) and valid on 1k (Crandon) with lstm but seq_len 1, with PTZ
# python train_nav.py --run 2k_ptz_lstm_1 --base_dir data/nav_train --train_env Superior --valid_env Crandon \
		# --train_dir nav0 --valid_dir nav0 --PTZ_weights weights/noise.pth --lr 0.001 --bsize 32 --seq_len 1 --with_lstm
