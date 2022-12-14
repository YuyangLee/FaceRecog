CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_50  --learnable_margin --t_ema --tag  50_lea --epoch 150 --aug
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_101 --learnable_margin --t_ema --tag 101_lea --epoch 200 --aug
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_50  --learnable_margin --t_ema --tag  50_le- --epoch 150
CUDA_VISIBLE_DEVICES=0 python run.py --backbone resnet_101 --learnable_margin --t_ema --tag 101_le- --epoch 200

