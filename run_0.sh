CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet_50 --loss triplet_l2 --margin_warmup --t_ema --tag  50_wea_trl2--aug --epoch 250
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet_50 --loss triplet_l2                 --t_ema --tag  50_-ea_trl2--aug --epoch 250
