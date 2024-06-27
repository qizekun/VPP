#!/bin/bash
gpu=$1
query="${@:2}"
CUDA_VISIBLE_DEVICES=$gpu python main.py --config cfgs/inference.yaml --inference --text --query "$query" --seed $RANDOM