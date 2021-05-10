#!/bin/bash

for i in Delton Goffs Oyens Placida Roane Springhill Sumas Woonsocket Superior Crandon
do
	python example.py --scene ../gibson/$i.glb --max_frames 50 --save_png --save_dir ../data/train/$i/nav0 --init_loc 0 --loc_num 2 --mode nav
	# python example.py --scene ../gibson/$i.glb --max_frames 50 --save_png --save_dir ../data/train/$i/nav1 --init_loc 100 --loc_num 100	--mode nav
done 