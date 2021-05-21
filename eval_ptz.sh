#!/bin/bash

python scripts/train_ptz.py --eval --overlap 1 --weight weights/noise.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/noise.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/habitat.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/habitat.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/noise.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/noise.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/fractal_tune_lep.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/fractal_tune_lep.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/nshape_tune_lep.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/nshape_tune_lep.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/perlin_tune_lep.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/perlin_tune_lep.pth

python scripts/train_ptz.py --eval --overlap 1 --weight weights/habitat_tune.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/habitat_tune.pth