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

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--H", default=64, type=int)
    parser.add_argument("--W", default=64, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    
    parser.add_argument("--margin", default=0.15, type=float)
    parser.add_argument("--learnable_margin", action="store_true")
    parser.add_argument("--t_ema", action="store_true")

    parser.add_argument("--backbone", default="resnet_50", type=str)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--tag", default="train", type=str)
    
    parser.add_argument("--test", action="store_true")
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
    acc = 1 - (tn + fp) / 2
    return tn, fp, acc

@torch.jit.script
def triplet_loss(anchor, positive, negative, margin):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    return loss

def train(args, basedir, model, train_dataset, valid_dataset, writer):
    params = [{ 'params': model.parameters(), 'lr': 1e-3 }]
    
    margin = torch.tensor(args.margin).to(args.device)
    if args.learnable_margin:
        margin.requires_grad_()
        params.append({ 'params': margin, 'lr': 1e-5 })
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(params)
    
    t_ema = EMA(0.99)
    flipper = RandomHorizontalFlip(p=1.0)
    rotater = RandomRotation(degrees=10)
    
    step = 0
    for epoch in range(args.epochs):
        tqdm.write(f"Epoch: { epoch }")
        model.train()
        for i_batch, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
            faces_anc, faces_pos, faces_neg = faces_anc.permute(0, 3, 1, 2), faces_pos.permute(0, 3, 1, 2), faces_neg.permute(0, 3, 1, 2)
            
            pos_flip = (idx_anc == idx_pos)
            pos_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5 + pos_flip
            neg_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5
            
            pos_flip_idx = torch.where(pos_flip)[0]
            neg_flip_idx = torch.where(neg_flip)[0]
            
            faces_pos[pos_flip_idx] = flipper(faces_pos[pos_flip_idx])
            faces_neg[neg_flip_idx] = flipper(faces_neg[neg_flip_idx])
            
            faces_anc = rotater(faces_anc)
            faces_pos = rotater(faces_pos)
            faces_neg = rotater(faces_neg)
            
            faces_anc, faces_pos, faces_neg = faces_anc.permute(0, 2, 3, 1), faces_pos.permute(0, 2, 3, 1), faces_neg.permute(0, 2, 3, 1)
            
            faces_anc = faces_anc + torch.normal(mean=0.0, std=0.02, size=faces_anc.shape, device=args.device)
            faces_pos = faces_pos + torch.normal(mean=0.0, std=0.02, size=faces_pos.shape, device=args.device)
            faces_neg = faces_neg + torch.normal(mean=0.0, std=0.02, size=faces_neg.shape, device=args.device)
            
            faces_anc, faces_pos, faces_neg = torch.clamp(faces_anc, 0, 1), torch.clamp(faces_pos, 0, 1), torch.clamp(faces_neg, 0, 1)
            
            ft_anc = model(faces_anc)
            ft_pos = model(faces_pos)
            ft_neg = model(faces_neg)
            
            loss_t = triplet_loss(ft_anc, ft_pos, ft_neg, margin=margin)
            loss_n = (torch.norm(ft_anc, dim=-1) - 1).pow(2).mean() + (torch.norm(ft_pos, dim=-1) - 1).pow(2).mean()  + (torch.norm(ft_neg, dim=-1) - 1).pow(2).mean() 
            loss_r = torch.tensor([p.pow(2.0).sum() for p in model.parameters()], device=args.device).mean()
 
            loss = 10 * loss_t.mean() + 0.01 * loss_n.mean() + 0.01 * loss_r
            loss.backward()
            optimizer.step()
            step += 1
            
            with torch.no_grad():
                if i_batch % 10 == 0:
                    threshold = (torch.norm(ft_pos - ft_anc, dim=-1).max() + torch.norm(ft_neg - ft_anc, dim=-1).min()) / 2
                    threshold = threshold.item()
                    
                    if args.t_ema:
                        t_ema.update(threshold)
                        threshold = t_ema.value()
                    
                    tn, fp, acc = metric(ft_anc, ft_pos, ft_neg, threshold=threshold)
                    writer.add_scalar("train/epoch", epoch, step)
                    
                    writer.add_scalar("train/t_neg", tn, step)
                    writer.add_scalar("train/f_pos", fp, step)
                    writer.add_scalar("train/accuracy", acc, step)
                    writer.add_scalar("train/triplet_loss", loss.item(), step)  
                      
                    writer.add_scalar("hparam/threshold", threshold, step)
                    writer.add_scalar("hparam/margin", margin, step)
                    
                    # writer.add_image("train/image/anc", faces_anc[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    # writer.add_image("train/image/pos", faces_pos[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    # writer.add_image("train/image/neg", faces_neg[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    
                    tqdm.write(f"\tLoss: {loss.item():.4f}, Margin: {margin:.4f}, Threshold: {threshold:.4f}, Accuracy: {acc:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
                
            
        with torch.no_grad():
            model.eval()
            tns, fps, accs = [], [], []
            for i_batch, data in enumerate(tqdm(valid_dl)):
                faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
                
                ft_anc = model(faces_anc)
                ft_pos = model(faces_pos)
                ft_neg = model(faces_neg)
                
                tn, fp, acc = metric(ft_anc, ft_pos, ft_neg, threshold=threshold)
                
                tns.append(tn)
                fps.append(fp)
                accs.append(acc)
                
            writer.add_scalar("valid/t_neg", np.array(tns).mean(), step)
            writer.add_scalar("valid/f_pos", np.array(fps).mean(), step)
            writer.add_scalar("valid/accuracy", np.array(accs).mean(), step)
        
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
    
    set_seed(args.seed)
    
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
        