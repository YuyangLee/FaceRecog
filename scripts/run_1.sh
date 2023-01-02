CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --loss liftstr --dist_metric l2 --margin_warmup --t_ema --tag  50_wea_lsl2 --aug --epoch 250 --num_workers 16
# CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --loss liftstr --dist_metric l2                 --t_ema --tag  50_-ea_lsl2 --aug --epoch 250 --num_workers 16
# CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --loss liftstr --dist_metric l2 --margin_warmup         --tag  50_w-a_lsl2 --aug --epoch 250 --num_workers 16
# CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet_50 --loss liftstr --dist_metric l2 --margin_warmup --t_ema --tag  50_we-_lsl2       --epoch 250 --num_workers 16
