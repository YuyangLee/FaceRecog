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
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data.Faces import Faces
from models.RecogNet import RecogNet
from utils.EMA import EMA
from utils.utils_dev import set_seed, get_args, has_nan
from utils.utils_loss import get_loss
import torch.multiprocessing as mp

t_ema = EMA(0.95)

def triplet_based_forward(args, model, data, aug_warpper, self_flipper, loss_fn, margin, require_image=False):
    faces_anc, faces_pos, faces_neg, idx_anc, idx_pos, idx_neg = data
    faces_anc, faces_pos, faces_neg = faces_anc.permute(0, 3, 1, 2), faces_pos.permute(0, 3, 1, 2), faces_neg.permute(0, 3, 1, 2)
    
    if args.aug:
        pos_flip = (idx_anc == idx_pos)
        pos_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5 + pos_flip
        neg_flip = torch.rand(pos_flip.shape, device=args.device) < 0.5
        
        pos_flip_idx = torch.where(pos_flip)[0]
        neg_flip_idx = torch.where(neg_flip)[0]
        
        faces_pos[pos_flip_idx] = self_flipper(faces_pos[pos_flip_idx])
        faces_neg[neg_flip_idx] = self_flipper(faces_neg[neg_flip_idx])
        
        faces_anc = aug_warpper(faces_anc)
        faces_pos = aug_warpper(faces_pos)
        faces_neg = aug_warpper(faces_neg)
        
        faces_anc, faces_pos, faces_neg = torch.clamp(faces_anc, 0, 1), torch.clamp(faces_pos, 0, 1), torch.clamp(faces_neg, 0, 1)
    
    ft_anc = model(faces_anc)
    ft_pos = model(faces_pos)
    ft_neg = model(faces_neg)
    
    loss_t, threshold, tn, fp = loss_fn(ft_anc, ft_pos, ft_neg, margin=margin)
    loss_n = (torch.norm(ft_anc, dim=-1) - 1).pow(2) + (torch.norm(ft_pos, dim=-1) - 1).pow(2) + (torch.norm(ft_neg, dim=-1) - 1).pow(2)
    loss_n = torch.clamp(loss_n, 0, 2.0)
    # loss_r = torch.tensor([p.pow(2.0).sum() for p in model.parameters()], device=args.device).mean()
    
    loss = 10 * loss_t.mean() + loss_n.mean() # + 0.001 * loss_r
    
    if t_ema is not None:
        t_ema.update(threshold.item())
        threshold = t_ema.value()
    
    res = {
        "loss": loss,
        "stat": {
            "hparam/threshold": threshold,
            "train/margin": margin,
            "train/triplet_loss": loss_t.mean(),
            "train/norm_loss": loss_n.mean(),
            "train/loss": loss,
            "train/tn": tn,
            "train/fp": fp,
        },
    }
    if require_image:
        res['images'] = {
            "train/image/anc": faces_anc[0].detach().cpu().numpy(),
            "train/image/pos": faces_pos[0].detach().cpu().numpy(),
            "train/image/neg": faces_neg[0].detach().cpu().numpy()
        }
    return res, threshold

def pair_based_forward(args, model, data, aug_warpper, self_flipper, loss_fn, margin, require_image=False):
    face_0, face_1, label_0, label_1, same_flag = data
    face_0, face_1 = face_0.permute(0, 3, 1, 2), face_1.permute(0, 3, 1, 2)
    
    if args.aug:
        flip = (torch.rand(label_0.shape, device=args.device) < 0.5) + same_flag
        flip = torch.where(flip)[0]
        
        face_1[flip] = self_flipper(face_1[flip])
        
        face_0 = torch.clamp(aug_warpper(face_0), 0., 1.)
        face_1 = torch.clamp(aug_warpper(face_1), 0., 1.)
    
    faces, labels = torch.cat([face_0, face_1], dim=0), torch.cat([label_0, label_1], dim=0)
    fts = model(faces)
    
    loss_p, threshold, tn, fp = loss_fn(fts, labels, margin=margin)
    loss_n = (torch.norm(fts, dim=-1) - 1).pow(2).mean()
    loss_n = torch.clamp(loss_n, 0, 2.0)
    # loss_r = torch.tensor([p.pow(2.0).sum() for p in model.parameters()], device=args.device).mean()
    
    loss = 10.0 * loss_p.mean() + loss_n.mean() # + 0.001 * loss_r
    
    if has_nan(loss):
        raise NotImplementedError()
    
    if t_ema is not None:
        t_ema.update(threshold.item())
        threshold = t_ema.value()
    
    res = {
        "loss": loss,
        "stat": {
            "hparam/threshold": threshold,
            "train/margin": margin,
            "train/pair_loss": loss_p.mean(),
            "train/norm_loss": loss_n.mean(),
            "train/loss": loss,
            "train/tn": tn,
            "train/fp": fp,
        }
    }
    
    if require_image:
        res['images'] = {
            "train/image/face_0": face_0[0].detach().cpu().numpy(),
            "train/image/face_1": face_1[0].detach().cpu().numpy(),
        }
    return res, threshold

def train(args, basedir, model, train_dataset, valid_dataset, frw_fn, loss_fn, writer):
    params = [{ 'params': model.parameters(), 'lr': 1e-3 }]
    
    margin = torch.tensor(args.margin).to(args.device)
    if args.margin_warmup:
        get_margin = lambda x: (min(2.0, 1 + x / args.margin_warmup_steps)) * args.margin / 2
    else:
        get_margin = lambda x: args.margin
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(params)
    
    flipper = RandomHorizontalFlip(p=1.0)
    aug_transforms = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(15, 1.0)], p=0.1),
        transforms.RandomAffine(degrees=10, scale=(0.95, 1.05), shear=5),
        transforms.RandomGrayscale(p=0.25),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
        # transforms.RandomErasing(scale=(0.02, 0.05)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=[0.2, 0.2, 0.2]),
    ])
    norm_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    ])

    aug_warpper = aug_transforms if args.aug else norm_transforms
    
    step = 0
    for epoch in range(args.epochs):
        tqdm.write(f"Epoch: { epoch }")
        
        margin = get_margin(step)
        model.train()
        for i_batch, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            data = [d.to(args.device) for d in data]
            
            require_image = i_batch == 0
            
            res, threshold = frw_fn(args, model, data, aug_warpper, flipper, loss_fn, margin, require_image)
            
            loss, stat = res['loss'], res['stat']
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            
            if i_batch % 10 == 0:
                for k, v in stat.items():
                    writer.add_scalar(k, v, step)
                
                tn, fp = res['stat']['train/tn'], res['stat']['train/fp']
                tqdm.write(f"\tLoss: {loss.item():.4f}, Margin: {margin:.4f}, Threshold: {threshold:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
                
            if require_image:
                for k, v in res['images'].items():
                    writer.add_image(k, v, step)
            step += 1
            
        if epoch % 5 == 4:
            with torch.no_grad():
                model.eval()
                tns, fps, accs = [], [], []
                labels = []
                for i_batch, data in enumerate(tqdm(valid_dl)):
                    data = [d.to(args.device) for d in data]
                    faces_0, faces_1, label = data
                    
                    faces_0, faces_1 = faces_0.permute(0, 3, 1, 2), faces_1.permute(0, 3, 1, 2)
                    faces_0, faces_1 = torch.clamp(norm_transforms(faces_0), 0., 1.), torch.clamp(norm_transforms(faces_1), 0., 1.)
                    
                    ft_0, ft_1 = model(faces_0), model(faces_1)
                    
                    dist = torch.norm(ft_0 - ft_1, dim=-1)
                    pred_1 = (dist < threshold).float()
                    pred_0 = 1 - pred_1
                    
                    tn = pred_0 * label
                    fp = pred_1 * (1 - label)
                    acc = ((pred_1 * label) + pred_0 * (1 - label))
                    
                    tns.append(tn)
                    fps.append(fp)
                    accs.append(acc)
                    labels.append(label)
                
                fps, tns, accs, labels = torch.cat(fps), torch.cat(tns), torch.cat(accs), torch.cat(labels)
                
                fp = fps.sum() / (1 - labels).sum()
                tn = tns.sum() / labels.sum()
                acc = accs.mean()
                
                writer.add_scalar("valid/f_pos", fp, step)
                writer.add_scalar("valid/t_neg", tn, step)
                writer.add_scalar("valid/accuracy", acc.mean(), step)
                tqdm.write(f"\tThreshold: {threshold:.4f}, Accuracy: {acc:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
        
        if epoch % 50 == 9:
            with torch.no_grad():
                torch.save({
                    "args": args,
                    "threshold": threshold,
                    "margin": margin,
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


if __name__ == '__main__':
    args = get_args()
    
    set_seed(args.seed)
    
    basedir = "results/" + datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + f"_{ args.tag }"
    os.makedirs(basedir, exist_ok=True)
    
    writer = tensorboardX.SummaryWriter(log_dir=basedir)
    recognet = RecogNet(args.H, args.W, len_embedding=256, backbone=args.backbone).to(args.device)
    
    if args.loss == 'triplet':
        frw_fn = triplet_based_forward
        train_extractor = 'triplet'
    elif args.loss == 'liftstr':
        frw_fn = pair_based_forward
        train_extractor = 'pair'
    elif args.loss == 'pairwise':
        frw_fn = pair_based_forward
        train_extractor = 'pair'
    else:
        raise NotImplementedError("Loss function not implemented")
    loss_fn = get_loss(args.loss, metric=args.dist_metric)
    
    if not args.test:
        with torch.no_grad():
            train_ds = Faces("data", args.batch_size, args.H, args.W, mode='train', train_extractor=train_extractor, lazy=args.lazy_load, device=args.preload_device)
            valid_ds = Faces("data", args.batch_size, args.H, args.W, mode='valid', train_extractor=train_extractor,lazy=args.lazy_load, device=args.preload_device)
        train(args, basedir, recognet, train_ds, valid_ds, frw_fn, loss_fn, writer)

    else:
        with torch.no_grad():
            test_ds = Faces("data", args.batch_size, mode='test', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        test(args, basedir, recognet, test_ds, writer)
        