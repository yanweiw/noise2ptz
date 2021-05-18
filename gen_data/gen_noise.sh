#!/bin/bash

python ../scripts/aug_data.py --type fractal --save_path ../data/noise_train/fractal --num 10000
python ../scripts/aug_data.py --type fractal --save_path ../data/noise_test/fractal --num 1000
python ../scripts/aug_data.py --type perlin --save_path ../data/noise_train/perlin --num 10000
python ../scripts/aug_data.py --type perlin --save_path ../data/noise_test/perlin --num 1000
python ../scripts/aug_data.py --type shape --save_path ../data/noise_train/nonoverlap --num 10000 
python ../scripts/aug_data.py --type shape --save_path ../data/noise_test/nonoverlap --num 1000 
python ../scripts/aug_data.py --type shape --save_path ../data/noise_train/overlap --num 10000 --overlap
python ../scripts/aug_data.py --type shape --save_path ../data/noise_test/overlap --num 1000 --overlap