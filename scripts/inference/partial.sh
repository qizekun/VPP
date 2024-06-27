#!/bin/bash
gpu=$1
query="${@:2}"
pts_path=$3
CUDA_VISIBLE_DEVICES=$gpu python main.py --config cfgs/inference.yaml --partial --query "$query" --pts_path $pts_path --seed $RANDOM
