#!/bin/bash

# python train_ptz.py --train --run noise --train_dir data/noise_train --test_dir data/noise_test --lr 0.001 --bsize 512 \
					# --epochs 5000 --parallel --overlap 0.666667

# python scripts/train_ptz.py --train --run fractal --train_dir data/fractal_train --test_dir data/fractal_test \
							# --lr 0.001 --bsize 512 --epochs 5000 --parallel --overlap 0.666667					

# python scripts/train_ptz.py --train --run overlap --train_dir data/shape_o_train --test_dir data/shape_o_test \
							# --lr 0.001 --bsize 512 --epochs 1500 --parallel --overlap 0.666667												

# python scripts/train_ptz.py --train --run perlin --train_dir data/perlin_train --test_dir data/perlin_test \
							# --lr 0.001 --bsize 512 --epochs 1500 --parallel --overlap 0.666667					

# python scripts/train_ptz.py --train --run nonoverlap --train_dir data/shape_n_train --test_dir data/shape_n_test \
							# --lr 0.001 --bsize 512 --epochs 1500 --parallel --overlap 0.666667

python scripts/train_ptz.py --train --run tmp --train_dir data/shape_n_train --test_dir data/shape_n_test \
							--lr 0.001 --bsize 512 --epochs 5000 --parallel --overlap 0.666667 --weight weights/nonoverlap.pth							

# python scripts/train_ptz.py --train --run fractal_cont --train_dir data/fractal_train --test_dir data/fractal_test \
							# --lr 0.001 --bsize 512 --epochs 5000 --parallel --overlap 0.666667 --weight weights/fractal.pth							