import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import tensorboardX
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
import cv2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(42)

device = 'cuda'
batch_size = 128

ckpt = "archive/2023-01-12/2023-01-12/15-50-06_cmp_r18_wa_trl2/resnet18_349.pth"
recognet = RecogNet(128, 128, len_embedding=256, backbone=os.path.basename(ckpt).split('_')[0]).to(device)
recognet.load_state_dict(torch.load(ckpt)['model'])

norm_transforms = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.2, 0.2, 0.2])
])

resizer = Resize((128, 128))
basedir = "data/search_imgs"

img_paths = [
    # "data/search_imgs/adele.png",
    # "data/search_imgs/bush.png",
    # "data/search_imgs/colin_powell.png",
    # "data/search_imgs/daniel_craig.png",
    # "data/search_imgs/rick_astley.png",
    "data/search_imgs/tom_cruise.png",
]
imgs = []
for img_path in img_paths:
    img = torch.tensor(cv2.imread(img_path, cv2.IMREAD_COLOR)).float()[:, :, [2, 1, 0]] / 256 # BGR to RGB and scale to [0, 1]
    imgs.append(resizer(img.permute(2, 0, 1)))
    
imgs = torch.stack(imgs, dim=0).to(device)
imgs = norm_transforms(imgs)
fts = recognet(imgs)

train_ds = Faces("data", batch_size, 128, 128, mode='train', train_extractor='single', lazy=False, device=device)
train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

dists = []
for i_batch, data in enumerate(tqdm(train_dl)):
    data = data.permute(0, 3, 1, 2)
    fts_db = recognet(data.to(device))
    dist = torch.norm(fts.unsqueeze(1).tile([1, data.shape[0], 1]) - fts_db.unsqueeze(0).tile([fts.shape[0], 1, 1]), dim=-1) # N x B
    dists.append(dist.detach().cpu())
    
dists = torch.cat(dists, dim=1) # N x N_train
match = torch.topk(-dists, dim=-1, k=4)[1].cuda()
dists = - torch.topk(-dists, dim=-1, k=4)[0].cuda()

for i in range(imgs.shape[0]):
    exp_dir = os.path.join(basedir, os.path.basename(img_paths[i])[:-4])
    os.makedirs(exp_dir, exist_ok=True)
    print(img_paths[i])
    print(dists[i])
    for j in range(4):
        img = train_ds.get_image(match[i][j]).permute(2, 0, 1)
        save_image(img, os.path.join(exp_dir, f"{str(j)}.jpg"))
