CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50  --learnable_margin         --tag  50_l-a --epoch 150 --aug
CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50                     --t_ema --tag  50_-ea --epoch 150 --aug
