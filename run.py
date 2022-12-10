import argparse
import json
import os
from datetime import datetime

import tensorboardX
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
from utils.EMA import EMA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--H", default=128, type=int)
    parser.add_argument("--W", default=128, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--learnable_margin", action="store_true")
    parser.add_argument("--t_ema", action="store_true")

    parser.add_argument("--backbone", default="resnet_101", type=str)
    parser.add_argument("--loss", default="triplet", type=str)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--tag", default="train", type=str)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    
    args = parser.parse_args()
    return args

# @torch.jit.script
def metric(anchor, positive, negative, threshold=0.1):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    tn = (dist_ap > threshold).float().mean().item()
    fp = (dist_an < threshold).float().mean().item()
    acc = tn + fp / 2
    return tn, fp, acc

# @torch.jit.script
def triplet_loss(anchor, positive, negative, margin=0.2):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    loss = torch.mean(torch.clamp(dist_ap - dist_an + margin, min=0.0))
    return loss

def train(args, basedir, model, train_dataset, valid_dataset, writer):
    params = [{ 'params': model.parameters() }]
    
    margin = torch.tensor(args.margin).to(args.device)
    if args.learnable_margin:
        margin.requires_grad_()
        params.append({ 'params': margin })
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(params)
    
    t_ema = EMA(0.99)
    flipper = RandomHorizontalFlip(p=1.0)
    rotater = RandomRotation(degrees=10)
    
    for epoch in range(args.epochs):
        tqdm.write(f"Epoch: { epoch }")
        model.train()
        for i_batch, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
            
            pos_flip = (idx_anc == idx_pos)
            pos_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5 + pos_flip
            neg_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5
            
            pos_flip_idx = torch.where(pos_flip)[0]
            neg_flip_idx = torch.where(neg_flip)[0]
            
            faces_pos[pos_flip_idx] = flipper(faces_pos[pos_flip_idx])
            faces_neg[neg_flip_idx] = flipper(faces_neg)[neg_flip_idx]
            
            ft_anc = model(rotater(faces_anc))
            ft_pos = model(rotater(faces_pos))
            ft_neg = model(rotater(faces_neg))
            
            loss = triplet_loss(ft_anc, ft_pos, ft_neg, margin=margin.item())
            loss.backward()
            optimizer.step()
            
            if i_batch % 10 == 0:
                with torch.no_grad():
                    threshold = (torch.norm(ft_pos - ft_anc, dim=-1).max() + torch.norm(ft_neg - ft_anc, dim=-1).min()) / 2
                    threshold = threshold.item()
                    
                    if args.t_ema:
                        t_ema.update(threshold)
                        threshold = t_ema.value()
                    
                    tn, fp, acc = metric(ft_anc, ft_pos, ft_neg, threshold=threshold)
                    writer.add_scalar("train/t_neg", tn, i_batch)
                    writer.add_scalar("train/f_pos", fp, i_batch)
                    writer.add_scalar("train/accuracy", acc, i_batch)
                    writer.add_scalar("train/triplet_loss", loss.item(), i_batch)    
                    
                    tqdm.write(f"\tLoss: {loss.item():.4f}, Margin: {margin:.4f}, Threshold: {threshold:.4f}, Accuracy: {acc:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
                
        with torch.no_grad():
            model.eval()
            for i_batch, data in enumerate(tqdm(valid_dl)):
                faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
                
                ft_anc = model(faces_anc)
                ft_pos = model(faces_pos)
                ft_neg = model(faces_neg)
                
                tn, fp, acc = metric(ft_anc, ft_pos, ft_neg, threshold=threshold)
                writer.add_scalar("valid/t_neg", tn, i_batch)
                writer.add_scalar("valid/f_pos", fp, i_batch)
                writer.add_scalar("valid/accuracy", acc, i_batch)
        
        with torch.no_grad():
            if epoch % 10 == 0:
                torch.save({
                    "args": args,
                    "threshold": threshold,
                    "margin": margin.item(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(basedir, f"{ args.backbone }_{ epoch }.pth"))

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
    margin = ckpt["margin"]
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return margin

if __name__ == '__main__':
    args = get_args()
    
    basedir = "results/" + datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + f"_{ args.tag }"
    os.makedirs(basedir, exist_ok=True)
    
    writer = tensorboardX.SummaryWriter(log_dir=basedir)
    recognet = RecogNet(args.H, args.W, len_embedding=256, backbone=args.backbone).to(args.device)
    
    if not args.test:
        with torch.no_grad():
            train_ds = Faces("data", args.batch_size, args.H, args.W, mode='train', train_ratio=args.train_ratio, lazy=args.lazy_load, preload_device='cuda', device='cuda')
            valid_ds = Faces("data", args.batch_size, args.H, args.W, mode='valid', train_ratio=args.train_ratio, lazy=args.lazy_load, preload_device='cuda', device='cuda')
        train(args, basedir, recognet, train_ds, valid_ds, writer)

    else:
        with torch.no_grad():
            test_ds = Faces("data", args.batch_size, mode='test', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        test(args, basedir, recognet, test_ds, writer)
        