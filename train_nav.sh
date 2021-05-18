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

## train on 1k (Superior) and valid on 1k (Crandon) with lstm but seq_len 1, with PTZ
# python train_nav.py --run 2k_ptz_lstm_1 --base_dir data/nav_train --train_env Superior --valid_env Crandon \
		# --train_dir nav0 --valid_dir nav0 --PTZ_weights weights/noise.pth --lr 0.001 --bsize 32 --seq_len 1 --with_lstm

# train on 20k (10 env) and valid on 10k (10 env) one batch has 400 datapoints (10 envs * 50 locations)
# python train_nav.py --run 30k_lstm_15 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 \
					# --lr 0.001 --bsize 128 --seq_len 15 --with_lstm --data_parallel --num_loc 40

# python train_nav.py --run 2k_ptz_lstm_15 --base_dir data/nav_train --train_env Superior --valid_env Crandon --num_epochs 2000 \
# 		--train_dir nav0 --valid_dir nav0 --PTZ_weights weights/noise.pth --lr 0.001 --bsize 32 --seq_len 15 --with_lstm

# python train_nav.py --run 2k_ptz_lstm_5 --base_dir data/nav_train --train_env Superior --valid_env Crandon --num_epochs 2000 \
# 		--train_dir nav0 --valid_dir nav0 --PTZ_weights weights/noise.pth --lr 0.001 --bsize 32 --seq_len 5 --with_lstm		

# python train_nav.py --run 2k_lstm_1 --base_dir data/nav_train --train_env Superior --valid_env Crandon --num_epochs 2000 \
# 					--train_dir nav0 --valid_dir nav0 --lr 0.001 --bsize 32 --seq_len 1 --with_lstm 

# python train_nav.py --run 2k_lstm_5 --base_dir data/nav_train --train_env Superior --valid_env Crandon --num_epochs 2000 \
# 					--train_dir nav0 --valid_dir nav0 --lr 0.001 --bsize 32 --seq_len 5 --with_lstm 					

# python train_nav.py --run 30k_lstm_5 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 --num_epochs 600 \
# 					--lr 0.001 --bsize 128 --seq_len 5 --with_lstm --data_parallel --num_loc 40		

# python train_nav.py --run 60k_lstm_5 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 --num_epochs 600 \
# 					--lr 0.001 --bsize 128 --seq_len 5 --with_lstm --data_parallel

# python train_nav.py --run 30k_lstm_1 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 --num_epochs 600 \
# 					--lr 0.001 --bsize 128 --seq_len 1 --with_lstm --data_parallel --num_loc 40		

# python train_nav.py --run 60k_lstm_1 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 --num_epochs 600 \
# 					--lr 0.001 --bsize 128 --seq_len 1 --with_lstm --data_parallel

# python scripts/train_nav.py --run 60k_ptz_lstm_15 --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 --num_epochs 600 \
					# --lr 0.001 --bsize 128 --seq_len 15 --with_lstm --data_parallel --PTZ_weights weights/noise.pth

# python scripts/train_nav.py --run 30k_lstm_15_cont --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 \
# 					--lr 0.001 --bsize 128 --seq_len 15 --with_lstm --data_parallel --num_loc 40 \
# 					--pretrained_weights weights/30k_lstm_15.pth

python scripts/train_nav.py --run 60k_lstm_1_cont --base_dir data/nav_train --train_dir nav1 --valid_dir nav0 \
					--lr 0.001 --bsize 512 --seq_len 1 --with_lstm --data_parallel \
					--pretrained_weights weights/60k_lstm_1.pth