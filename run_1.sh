CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --margin_warmup --t_ema --tag  50_wea_trl2 --loss triplet_l2  --aug --epoch 150
CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50                 --t_ema --tag  50_-ea_trl2 --loss triplet_l2  --aug --epoch 150
CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --margin_warmup --t_ema --tag  50_wea_trcs --loss triplet_cos --aug --epoch 150
CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50                 --t_ema --tag  50_-ea_trcs --loss triplet_cos --aug --epoch 150
