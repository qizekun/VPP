#!/bin/bash
gpu=$1
img_path=$2
CUDA_VISIBLE_DEVICES=$gpu python main.py --config cfgs/inference.yaml --inference --img --img_path $img_path --seed $RANDOM