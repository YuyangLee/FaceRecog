# RecogNet

This is Yuyang's repository for his course project in PRML (THU Fall 2022).

## Environment Setup

## Data Preparation

Download the [dataset zip file](https://assets.aidenli.net/dev/thu-prml-22/dataset.zip), unzip it in `data/`. The directory should be like:

```
data/
├─test_pair/
│  ├─0/
│  └─...
└─training_set/
   ├─Alice/
   ├─Bob/
   └─...
```

Then pre-process the image:

```shell
python data-proc.py --dataset_dir data/
```

## Train the model

Specify the training parameters in `config/hp.yaml`, then run:

```shell
python train.py
```

We also provide a checkpoint [recognet_demo_ckpt.pt (PENDING)](#). Download it and put it to `ckpt/recognet_demo_ckpt.pt`.

## Test the model

Specify the hyperparameters in `config/hp.yaml`, then run:

```shell
python test.py --ckpt_path config/recognet_demo_ckpt.pt
```
