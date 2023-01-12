# CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone vgg11_bn --loss liftstr --dist_metric l2  --margin 0.6 --margin_warmup --tag cmp_v11_wa_lsl2 --aug --epoch 300 --num_workers 16
# CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone vgg11_bn --loss triplet --dist_metric l2  --margin 0.6 --margin_warmup --tag cmp_v11_wa_trl2 --aug --epoch 300 --num_workers 16
# CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet50 --loss liftstr --dist_metric l2  --margin 0.6 --margin_warmup --tag cmp_r34_wa_lsl2 --aug --epoch 300 --num_workers 16
# CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet50 --loss triplet --dist_metric l2  --margin 0.6 --margin_warmup --tag cmp_r34_wa_trl2 --aug --epoch 300 --num_workers 16

CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet34 --loss liftstr --dist_metric l2  --margin 0.6 --margin_warmup --tag abl_rsn_w-_trl2       --epoch 300 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet34 --loss liftstr --dist_metric l2  --margin 0.6                 --tag abl_rsn_-a_trl2 --aug --epoch 300 --num_workers 16
