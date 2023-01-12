# RecogNet

[Yuyang](https://yuyangli.com)'s repo for his PRML (THU Fall 22) course project.

### TL;DR

### 安装依赖

安装必要的依赖：

```shell
pip install matplotlib numpy opencv-python pandas seaborn scikit-learn tensorboardX torch torchvision tqdm 
```

并参考 [GitHub - DLib](https://github.com/davisking/dlib.git) 安装 `dlib`。

### 准备数据

下载 [数据集文件](https://assets.aidenli.net/dev/thu-prml-22/dataset.zip)，解压到 `data/`。目录结构如：


```
data/
├─test_pair/
│  ├─bb.json
│  ├─0/
│  └─...
└─training_set/
│  ├─bb.json
   ├─Alice/
   ├─Bob/
   └─...
```

### 训练模型

```shell
python 1_train.py
```

### 测试模型

我们提供一个训练好的 [ResNet34 的 checkpoint](https://assets.aidenli.net/dev/thu-prml-22/resnet34_release.pth)。

```shell
python 1_test.py --checkpoint [PATH_TO_CHECKPOINT]
```

### 测试数据预测结果

见 `test_res.txt`

## Environment Setup

### Install dependencies from pip

```shell
pip install -r requirements.txt
```

### `dlib` (required if you run data preprocessing)

We utilize `dlib` for face detection in data preprocessing.

#### Build from Source

First, `cd` into `thirdparty/`, and clone the repository:

```shell
git clone https://github.com/davisking/dlib.git
```
Then, `cd` into `dlib/`, and build from source:

```shell
mkdir build
cd build
cmake .. -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --set USE_AVX_INSTRUCTIONS=1
```

> If you have a CUDA compatible GPU, you can build with GPU support:
>
>```shell
>mkdir build
>cd build
>cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
>cmake --build .
>cd ..
>python setup.py install --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1
>```
> This requires CUDA and cuDNN library.

### Download Face Landmark Predictor

Download [predictor](https://assets.aidenli.net/dev/thu-prml-22/shape_predictor_68_face_landmarks.dat) into `ckpt/`


## Data Preparation (Optional)

Download the [dataset zip file](https://assets.aidenli.net/dev/thu-prml-22/dataset.zip), unzip it in `data/`. The directory should be like:

```
data/
├─test_pair/
│  ├─bb.json
│  ├─0/
│  └─...
└─training_set/
│  ├─bb.json
   ├─Alice/
   ├─Bob/
   └─...
```

### Image Pre-processing

To help training, we first pre-process the image:

- rotate the image, so that the eyes are on a horizontal line.
- detect the face and mark the bounding box

Pre-process the image:

```shell
python data-proc.py --dataset_dir data/
```

This will generate 2 bounding boxes for each image: one for the untransformed version, and another one for the transformed version. Parameters for the  affine tranformation as well as the bounding boxes are in `data/training_set/bb.json` or `data/test_pair/bb.json`.

> On our provided images, at least one face is (will be) detectable with our code. If you want to use your images, you may need to check that there is no failure cases, or you should optimize the detection code or manually annotate the data.

## Train the model

Specify the training parameters in `config/hp.yaml`, then run:

```shell
python 1_train.py
```

We also provide a checkpoint [recognet_demo_ckpt.pt (PENDING)](#). Download it and put it to `ckpt/recognet_demo_ckpt.pt`.

## Test the model

Specify the hyperparameters in `config/hp.yaml`, then run:

```shell
python 1_test.py --checkpoint [PATH_TO_CHECKPOINT]
```
