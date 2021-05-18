#!/bin/bash

python scripts/train_ptz.py --eval --overlap 1 --weight weights/noise.pth
python scripts/train_ptz.py --eval --overlap 0 --weight weights/noise.pth