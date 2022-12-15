import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import tensorboardX
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
from utils.EMA import EMA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=180, type=int)

    parser.add_argument("--H", default=128, type=int)
    parser.add_argument("--W", default=128, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    
    parser.add_argument("--margin", default=0.20, type=float)
    parser.add_argument("--margin_warmup", action="store_true")
    parser.add_argument("--margin_warmup_steps", default=5000, type=int)
    parser.add_argument("--t_ema", action="store_true")
    parser.add_argument("--aug", action="store_true")
    
    parser.add_argument("--max_grad_norm", default=5.0, type=float)

    parser.add_argument("--loss", default="triplet_l2", type=str)
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
    
def train(args, basedir, model, train_dataset, valid_dataset, loss_fn, writer):
    params = [{ 'params': model.parameters(), 'lr': 1e-3 }]
    
    margin = torch.tensor(args.margin).to(args.device)
    if args.margin_warmup:
        get_margin = lambda x: min(1.0, x / args.margin_warmup_steps) * args.margin
    else:
        get_margin = lambda x: args.margin
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(params)
    
    t_ema = EMA(0.99)
    flipper = RandomHorizontalFlip(p=1.0)
    
    if args.aug:
        aug_warpper =  transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomErasing(scale=(0.02, 0.16))
        ])
    else:
        aug_warpper = lambda x: x
    
    step = 0
    for epoch in range(args.epochs):
        tqdm.write(f"Epoch: { epoch }")
        
        margin = torch.tensor(get_margin(step), device=args.device, dtype=torch.float32)
        model.train()
        for i_batch, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
            faces_anc, faces_pos, faces_neg = faces_anc.permute(0, 3, 1, 2), faces_pos.permute(0, 3, 1, 2), faces_neg.permute(0, 3, 1, 2)
            
            if args.aug:
                pos_flip = (idx_anc == idx_pos)
                pos_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5 + pos_flip
                neg_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5
                
                pos_flip_idx = torch.where(pos_flip)[0]
                neg_flip_idx = torch.where(neg_flip)[0]
                
                faces_pos[pos_flip_idx] = flipper(faces_pos[pos_flip_idx])
                faces_neg[neg_flip_idx] = flipper(faces_neg[neg_flip_idx])
                
                faces_anc = aug_warpper(faces_anc)
                faces_pos = aug_warpper(faces_pos)
                faces_neg = aug_warpper(faces_neg)
                
                # faces_anc = faces_anc + torch.normal(mean=0.0, std=0.02, size=faces_anc.shape, device=args.device)
                # faces_pos = faces_pos + torch.normal(mean=0.0, std=0.02, size=faces_pos.shape, device=args.device)
                # faces_neg = faces_neg + torch.normal(mean=0.0, std=0.02, size=faces_neg.shape, device=args.device)
                
                faces_anc, faces_pos, faces_neg = torch.clamp(faces_anc, 0, 1), torch.clamp(faces_pos, 0, 1), torch.clamp(faces_neg, 0, 1)
            
            ft_anc = model(faces_anc)
            ft_pos = model(faces_pos)
            ft_neg = model(faces_neg)
            
            loss_t = loss_fn(ft_anc, ft_pos, ft_neg, margin=margin)
            loss_n = (torch.norm(ft_anc, dim=-1) - 1).pow(2).mean() + (torch.norm(ft_pos, dim=-1) - 1).pow(2).mean()  + (torch.norm(ft_neg, dim=-1) - 1).pow(2).mean() 
            # loss_r = torch.tensor([p.pow(2.0).sum() for p in model.parameters()], device=args.device).mean()
 
            loss = 10 * loss_t.mean() + loss_n.mean() # + 0.001 * loss_r
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            
            with torch.no_grad():
                if i_batch % 10 == 0:
                    threshold = (torch.norm(ft_pos - ft_anc, dim=-1).max() + torch.norm(ft_neg - ft_anc, dim=-1).min()) / 2
                    
                    if args.t_ema:
                        t_ema.update(threshold.item())
                        threshold = t_ema.value()
                    
                    dist_ap = torch.norm(ft_anc - ft_pos, dim=-1)
                    dist_an = torch.norm(ft_anc - ft_neg, dim=-1)
                    tn = (dist_ap > threshold).float().mean().item()
                    fp = (dist_an < threshold).float().mean().item()
                    
                    writer.add_scalar("train/t_neg", tn, step)
                    writer.add_scalar("train/f_pos", fp, step)
                    writer.add_scalar("train/triplet_loss", loss_t.mean().item(), step)  
                    writer.add_scalar("train/norm_loss", loss_n.mean().item(), step)  
                    # writer.add_scalar("train/reg_loss", loss_r.item(), step)  
                    writer.add_scalar("train/loss", loss.item(), step)  
                      
                    writer.add_scalar("hparam/threshold", threshold, step)
                    writer.add_scalar("hparam/margin", margin, step)
                    
                if i_batch % 50 == 0:
                    writer.add_image("train/image/anc", faces_anc[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    writer.add_image("train/image/pos", faces_pos[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    writer.add_image("train/image/neg", faces_neg[0].detach().cpu().numpy().transpose((2, 0, 1)), step)
                    
                    tqdm.write(f"\tLoss: {loss.item():.4f}, Margin: {margin:.4f}, Threshold: {threshold:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
                
            step += 1
            
        if epoch % 5 == 4:
            with torch.no_grad():
                model.eval()
                tns, fps, accs = [], [], []
                for i_batch, data in enumerate(tqdm(valid_dl)):
                    faces_0, faces_1, label = data
                    
                    faces_0, faces_1 = faces_0.permute(0, 3, 1, 2), faces_1.permute(0, 3, 1, 2)
                    
                    ft_0 = model(faces_0)
                    ft_1 = model(faces_1)
                    
                    dist = torch.norm(ft_0 - ft_1, dim=-1)
                    pred_1 = (dist < threshold).float()
                    pred_0 = 1 - pred_1
                    
                    fp = (pred_0 * label).sum().item() / (label.sum().item() + 1e-12)
                    tn = (pred_1 * (1 - label)).sum().item() / ((1 - label).sum().item() + 1e-12)
                    acc = ((pred_1 * label) + pred_0 * (1 - label)).sum().item() / args.batch_size
                    
                    tns.append(tn)
                    fps.append(fp)
                    accs.append(acc)
                    
                threshold = np.array(tns).mean()
                fp = np.array(fps).mean()
                tn = np.array(tns).mean()
                acc = np.array(accs).mean()
                
                writer.add_scalar("valid/f_pos", fp, step)
                writer.add_scalar("valid/t_neg", tn, step)
                writer.add_scalar("valid/accuracy", acc.mean(), step)
                tqdm.write(f"\tThreshold: {threshold:.4f}, Accuracy: {acc:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
        
        if epoch % 10 == 9:
            with torch.no_grad():
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
    threshold = ckpt["threshold"]
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return threshold

@torch.jit.script
def triplet_l2(anchor, positive, negative, margin):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    return loss

@torch.jit.script
def triplet_cos(anchor, positive, negative, margin):
    dist_ap = torch.cosine_similarity(anchor, positive, dim=-1)
    dist_an = torch.cosine_similarity(anchor, negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    return loss

def lifted_structure(features, labels):
    pass

if __name__ == '__main__':
    args = get_args()
    
    set_seed(args.seed)
    
    basedir = "results/" + datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + f"_{ args.tag }"
    os.makedirs(basedir, exist_ok=True)
    
    writer = tensorboardX.SummaryWriter(log_dir=basedir)
    recognet = RecogNet(args.H, args.W, len_embedding=256, backbone=args.backbone).to(args.device)
    
    if args.loss == 'triplet_l2':
        loss_fn = triplet_l2
    elif args.loss == 'triplet_cos':
        loss_fn = triplet_cos
    elif args.loss == 'nce':
        pass
    elif args.loss == 'infonce':
        pass
    else:
        raise NotImplementedError("Loss function not implemented")
    
    if not args.test:
        with torch.no_grad():
            train_ds = Faces("data", args.batch_size, args.H, args.W, mode='train', lazy=args.lazy_load, preload_device='cuda', device='cuda')
            valid_ds = Faces("data", args.batch_size, args.H, args.W, mode='valid', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        train(args, basedir, recognet, train_ds, valid_ds, loss_fn, writer)

    else:
        with torch.no_grad():
            test_ds = Faces("data", args.batch_size, mode='test', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        test(args, basedir, recognet, test_ds, writer)
        