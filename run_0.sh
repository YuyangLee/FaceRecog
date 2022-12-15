CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet_50  --learnable_margin --t_ema --tag  50_lea --epoch 150 --aug
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet_50  --learnable_margin --t_ema --tag  50_le- --epoch 150

