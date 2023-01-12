import argparse
import random

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=250, type=int)

    parser.add_argument("--H", default=128, type=int)
    parser.add_argument("--W", default=128, type=int)
    
    # Dataset
    parser.add_argument("--lazy_load", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    
    parser.add_argument("--margin", default=0.2, type=float)
    parser.add_argument("--margin_warmup", action="store_true")
    parser.add_argument("--margin_warmup_steps", default=2000, type=int)
    parser.add_argument("--t_ema", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--no_fnl", action="store_true")
    
    parser.add_argument("--max_grad_norm", default=5.0, type=float)

    parser.add_argument("--loss", default="liftstr", type=str)
    parser.add_argument("--dist_metric", default="l2", type=str)
    parser.add_argument("--backbone", default="resnet18", type=str)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--tag", default="train", type=str)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--preload_device", default="cpu", type=str)
    
    parser.add_argument("--seed", default=42, type=int)
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def has_nan(x):
    return torch.isnan(x).any()
    