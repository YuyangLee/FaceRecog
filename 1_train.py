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
from utils.utils_dev import set_seed, get_args
from utils.utils_loss import triplet_l2, triplet_cos, pairwise_l2, liftstr_l2

t_ema = EMA(0.95)

def triplet_based_forward(args, model, data, aug_warpper, self_flipper, loss_fn, margin):
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
        "images": {
            "train/image/anc": faces_anc[0].detach().cpu().numpy(),
            "train/image/pos": faces_pos[0].detach().cpu().numpy(),
            "train/image/neg": faces_neg[0].detach().cpu().numpy()
        }
    }
    return res, threshold

def pair_based_forward(args, model, data, aug_warpper, self_flipper, loss_fn, margin):
    face_0, face_1, label_0, label_1 = data
    face_0, face_1 = face_0.permute(0, 3, 1, 2), face_1.permute(0, 3, 1, 2)
    
    if args.aug:
        flip = (torch.rand(label_0.shape, device=args.device) < 0.5) + (label_0 == label_1)
        flip = torch.where(flip)[0]
        
        face_1[flip] = self_flipper(face_1[flip])
        
        face_0 = torch.clamp(aug_warpper(face_0), 0., 1.)
        face_1 = torch.clamp(aug_warpper(face_1), 0., 1.)
    
    faces, labels = torch.cat([face_0, face_1], dim=0), torch.cat([label_0, label_1], dim=0)
    fts = model(faces)
    
    loss_p, threshold, tn, fp = loss_fn(fts, labels, margin=margin)
    loss_n = (torch.norm(fts, dim=-1) - 1).pow(2).mean()
    # loss_r = torch.tensor([p.pow(2.0).sum() for p in model.parameters()], device=args.device).mean()
    
    loss = 1.0 * loss_p.mean() + 1.0 * loss_n.mean() # + 0.001 * loss_r
    
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
        },
        "images": {
            "train/image/face_0": face_0[0].detach().cpu().numpy(),
            "train/image/face_1": face_1[0].detach().cpu().numpy(),
        }
    }
    return res, threshold

def train(args, basedir, model, train_dataset, valid_dataset, frw_fn, loss_fn, writer):
    params = [{ 'params': model.parameters(), 'lr': 1e-3 }]
    
    margin = torch.tensor(args.margin).to(args.device)
    if args.margin_warmup:
        get_margin = lambda x: min(1.0, x / args.margin_warmup_steps) * args.margin
    else:
        get_margin = lambda x: args.margin
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(params)
    
    flipper = RandomHorizontalFlip(p=1.0)
    aug_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # transforms.RandomErasing(scale=(0.02, 0.05))
    ])

    aug_warpper = aug_transforms if args.aug else lambda x: x
    
    step = 0
    for epoch in range(args.epochs):
        tqdm.write(f"Epoch: { epoch }")
        
        margin = get_margin(step)
        model.train()
        for i_batch, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            
            res, threshold = frw_fn(args, model, data, aug_warpper, flipper, loss_fn, margin)
            
            loss, stat = res['loss'], res['stat']
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            
            if i_batch % 10 == 0:
                
                for k, v in stat.items():
                    writer.add_scalar(k, v, step)
                
            if i_batch % 50 == 0:
                # writer.add_image("train/image/anc", faces_anc[0].detach().cpu().numpy(), step)
                # writer.add_image("train/image/pos", faces_pos[0].detach().cpu().numpy(), step)
                # writer.add_image("train/image/neg", faces_neg[0].detach().cpu().numpy(), step)
                tn, fp = res['stat']['train/tn'], res['stat']['train/fp']
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
                    
                    fp = (pred_0 * label).sum() / (label.sum() + 1e-12)
                    tn = (pred_1 * (1 - label)).sum() / ((1 - label).sum() + 1e-12)
                    acc = ((pred_1 * label) + pred_0 * (1 - label)).sum() / args.batch_size
                    
                    tns.append(tn)
                    fps.append(fp)
                    accs.append(acc)
                    
                threshold = np.array(tns).mean()
                fp, tn, acc = np.array(fps).mean(), np.array(tns).mean(), np.array(accs).mean()
                
                writer.add_scalar("valid/f_pos", fp, step)
                writer.add_scalar("valid/t_neg", tn, step)
                writer.add_scalar("valid/accuracy", acc.mean(), step)
                tqdm.write(f"\tThreshold: {threshold:.4f}, Accuracy: {acc:.4f}, tn: {tn:.4f}, fp: {fp:.4f}")
        
        if epoch % 10 == 9:
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
        frw_fn = triplet_based_forward
        loss_fn = triplet_l2
        train_extractor = 'triplet'
    elif args.loss == 'triplet_cos':
        frw_fn = triplet_based_forward
        loss_fn = triplet_cos
        train_extractor = 'triplet'
    elif args.loss == 'liftstr_l2':
        frw_fn = pair_based_forward
        loss_fn = liftstr_l2
        train_extractor = 'pair'
    elif args.loss == 'pairwise':
        frw_fn = pair_based_forward
        loss_fn = pairwise_l2
        train_extractor = 'pair'
    else:
        raise NotImplementedError("Loss function not implemented")
    
    if not args.test:
        with torch.no_grad():
            train_ds = Faces("data", args.batch_size, args.H, args.W, mode='train', train_extractor=train_extractor, lazy=args.lazy_load, preload_device='cuda', device='cuda')
            valid_ds = Faces("data", args.batch_size, args.H, args.W, mode='valid', train_extractor=train_extractor,lazy=args.lazy_load, preload_device='cuda', device='cuda')
        train(args, basedir, recognet, train_ds, valid_ds, frw_fn, loss_fn, writer)

    else:
        with torch.no_grad():
            test_ds = Faces("data", args.batch_size, mode='test', lazy=args.lazy_load, preload_device='cuda', device='cuda')
        test(args, basedir, recognet, test_ds, writer)
        