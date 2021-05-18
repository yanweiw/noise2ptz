#!/bin/bash

# python scripts/eval.py --infer --deg 360 --policy 60k_ptz_lstm_15

python scripts/eval.py --plot --deg 360 --save_path figs --policy 2k_ptz_lstm_1  \
																  2k_ptz_lstm_5  \
																  2k_ptz_lstm_15 \
																  2k_lstm_1      \
																  2k_lstm_5      \
																  2k_lstm_15     \
																  30k_lstm_1     \
																  30k_lstm_5     \
																  30k_lstm_15    \
																  60k_lstm_1     \
																  60k_lstm_5     \
																  60k_lstm_15    
