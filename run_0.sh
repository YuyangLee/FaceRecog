# CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_50  --learnable_margin --t_ema --tag  50_l-e
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_50  --learnable_margin         --tag  50_l-x
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_101 --learnable_margin --t_ema --tag 101_l-e
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_101 --learnable_margin         --tag 101_l-x
