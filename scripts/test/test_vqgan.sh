CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/vqgan.yaml --exp_name $2 --vqgan --test --ckpts $3
