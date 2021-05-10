#!/bin/bash

while read env
do
	python example.py --scene ../gibson/$env.glb --max_frames 10 --save_png --save_dir ../data/tmp/$i/ --init_loc 0 --loc_num 10	
done