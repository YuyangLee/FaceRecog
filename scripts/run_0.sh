CUDA_VISIBLE_DEVICES=1 python 1_train.py --backbone resnet18 --loss triplet --dist_metric cos  --margin 0.6 --tag rsn_-a_trcd --aug --epoch 201 --num_workers 16 --no_fnl
