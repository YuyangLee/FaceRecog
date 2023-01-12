import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
from utils.EMA import EMA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=256, type=int)

    parser.add_argument("--H", default=128, type=int)
    parser.add_argument("--W", default=128, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    
    parser.add_argument("--dist", default="l2", type=str)
    parser.add_argument("--checkpoint", default="results/2023-01-12/18-26-15_cmp_r34_wa_trl2/resnet34_299.pth", type=str)
    # parser.add_argument("--backbone", default="vgg11_bn", type=str)
    parser.add_argument("--backbone", default="resnet34", type=str)
    parser.add_argument("--tag", default="test", type=str)
    
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

@torch.no_grad() 
def test(args, basedir, model, test_ds):
    threshold = load_ckpt(args.checkpoint, model=model, optimizer=None)
    model.eval()
    threshold = torch.tensor(threshold).to(args.device)
    
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)
    norm_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # for thres in [4.7, 4.8, 4.9, 5.0, threshold]:
    thres = threshold
    manifold_dist, recog_res = [], []
    for i_batch, data in enumerate(tqdm(test_dl)):
        data = [d.to(args.device) for d in data]
        faces_0, faces_1 = data
        
        export_dir = "results/test"
        faces_0, faces_1 = faces_0.permute(0, 3, 1, 2), faces_1.permute(0, 3, 1, 2)
        faces_0, faces_1 = norm_transforms(faces_0), norm_transforms(faces_1)
        
        if i_batch == 0:
            for i in range(16):
                save_image(faces_0[i] * 0.5 + 0.5, os.path.join(export_dir, f"{i}_A.jpg"))
                save_image(faces_1[i] * 0.5 + 0.5, os.path.join(export_dir, f"{i}_B.jpg"))
        
        ft_0 = model(faces_0)
        ft_1 = model(faces_1)
        
        if args.dist == "l2":
            dist = torch.norm(ft_0 - ft_1, dim=1)
        elif args.dist == "cos":
            dist = 1 - torch.cosine_similarity(ft_0, ft_1)
        else:
            raise NotImplementedError()
        
        manifold_dist += dist.cpu().numpy().tolist()
        recog_res += (dist < thres).int().cpu().numpy().tolist()
            
    # DEBUG
    with open("data/test_label.txt", 'r') as f:
        res = f.readlines()
        res = np.array([ int(r[0]) for r in res ])
        f.close()
    res = (np.array(recog_res) == res).mean()
    fp = (np.array(recog_res) * (1 - res)).mean()
    fn = ((1 - np.array(recog_res)) * res).mean()
    print("acc: " + str(res))
    print("fp: " + str(fp))
    print("fn: " + str(fn))
    # DEBUG
    
    recog_res = "\n".join([str(r) for r in recog_res])
    manifold_dist = "\n".join([str(d) for d in manifold_dist])
        
    with open(os.path.join(basedir, "res.txt"), 'w') as f:
        f.writelines(recog_res)
        f.close()
        
    with open(os.path.join(basedir, "dist.txt"), 'w') as f:
        f.writelines(manifold_dist)
        f.close()
        
def load_ckpt(path, model, optimizer=None):
    ckpt = torch.load(path)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    threshold = ckpt["threshold"]
    return threshold

if __name__ == '__main__':
    args = get_args()
    
    set_seed(args.seed)
    
    basedir = "results/" + datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + f"_{ args.tag }"
    os.makedirs(basedir, exist_ok=True)
    
    recognet = RecogNet(args.H, args.W, len_embedding=256, backbone=args.backbone).to(args.device)
    
    with torch.no_grad():
        test_ds = Faces("data", args.batch_size, H=args.H, W=args.W, mode='test', lazy=args.lazy_load, device='cuda')
    test(args, basedir, recognet, test_ds)
        