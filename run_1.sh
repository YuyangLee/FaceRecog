CUDA_VISIBLE_DEVICES=1 python run.py --backbone resnet_50  --learnable_margin         --tag  50_l-a --epoch 150 --aug
CUDA_VISIBLE_DEVICES=1 python run.py --backbone resnet_101 --learnable_margin         --tag 101_l-a --epoch 200 --aug
CUDA_VISIBLE_DEVICES=1 python run.py --backbone resnet_50                     --t_ema --tag  50_-ea --epoch 150 --aug
CUDA_VISIBLE_DEVICES=1 python run.py --backbone resnet_101                    --t_ema --tag 101_-ea --epoch 200 --aug
