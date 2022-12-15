import argparse
import json
import os
from datetime import datetime
import random

import numpy as np
import tensorboardX
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
from utils.EMA import EMA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=180, type=int)

    parser.add_argument("--H", default=128, type=int)
    parser.add_argument("--W", default=128, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    
    parser.add_argument("--margin", default=0.15, type=float)
    parser.add_argument("--learnable_margin", action="store_true")
    parser.add_argument("--t_ema", action="store_true")
    parser.add_argument("--aug", action="store_true")

    parser.add_argument("--backbone", default="resnet_50", type=str)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--tag", default="train", type=str)
    
    parser.add_argument("--test", default=True, type=bool)
    parser.add_argument("--device", default="cuda", type=str)
    
    parser.add_argument("--seed", default=42, type=int)
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def metric(anchor, positive, negative, threshold):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    tn = (dist_ap > threshold).float().mean().item()
    fp = (dist_an < threshold).float().mean().item()
    return tn, fp

@torch.jit.script
def triplet_loss(anchor, positive, negative, margin):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    return loss

@torch.no_grad() 
def test(args, basedir, model, test_ds, writer):
    model.eval()
    margin = load_ckpt(args.checkpoint, model=model, optimizer=None)
    margin = torch.tensor(margin).to(args.device)
    
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)
    
    manifold_dist, recog_res = [], []
    for i_batch, data in enumerate(tqdm(test_dl)):
        img_0, img_1 = data
        ft_0 = model(img_0)
        ft_1 = model(img_1)
        dist = torch.norm(ft_0 - ft_1, dim=1)
        manifold_dist += dist.cpu().numpy().tolist()
        recog_res += (dist < margin).cpu().numpy().tolist()
        
    with open(os.path.join(basedir, "res.txt", 'w')) as f:
        f.truncate()
        f.writelines(recog_res)
        f.close()
        
    with open(os.path.join(basedir, "dist.txt", 'w')) as f:
        f.truncate()
        f.writelines(manifold_dist)
        f.close()
        
def load_ckpt(path, model, optimizer=None):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    threshold = ckpt["threshold"]
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return threshold

if __name__ == '__main__':
    args = get_args()
    
    set_seed(args.seed)
    
    basedir = "results/" + datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + f"_{ args.tag }"
    os.makedirs(basedir, exist_ok=True)
    
    writer = tensorboardX.SummaryWriter(log_dir=basedir)
    
    recognet = RecogNet(args.H, args.W, len_embedding=256, backbone=args.backbone).to(args.device)
    threshold = load_ckpt(args.checkpoint, model=recognet, optimizer=None)
    
    if not args.test:
        pass
    
    else:
        with torch.no_grad():
            test_ds = Faces("data", args.batch_size, mode='test', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        test(args, basedir, recognet, test_ds, writer)
        