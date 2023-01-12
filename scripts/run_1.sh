CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss liftstr      --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_lrl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss triplet      --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_trl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss triplet_weak --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_wtl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone vgg11_bn --loss liftstr      --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_lrl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone vgg11_bn --loss triplet      --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_trl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet34 --loss liftstr      --dist_metric l2  --margin 0.4 --margin_warmup --tag cmp_rsn_wa_lrl2 --aug --epoch 201 --num_workers 16

CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss liftstr --dist_metric l2  --margin 0.4 --margin_warmup --tag abl_rsn_wa_trl2 --aug --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss liftstr --dist_metric l2  --margin 0.4 --margin_warmup --tag abl_rsn_w-_trl2       --epoch 201 --num_workers 16
CUDA_VISIBLE_DEVICES=0 python 1_train.py --backbone resnet18 --loss liftstr --dist_metric l2  --margin 0.4                 --tag abl_rsn_-a_trl2 --aug --epoch 201 --num_workers 16
