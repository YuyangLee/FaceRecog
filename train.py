import torch
from torch.utils.data import DataLoader
from data.Faces import Faces
from models.RecogNet import RecogNet
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--train_ratio", default=0.7, type=float)

    parser.add_argument("--backbone", default="resnet_50", type=str)
    parser.add_argument("--loss", default="triplet", type=str)

    args = parser.get_args()
    return args

def train(args, model, dataset):
    
    pass

if __name__ == '__main__':
    args = get_args()

    dataset = Faces("data/training_set", args.batch_size, mode='train', lazy=False, preload_device='cuda', device='cuda')

    recognet = RecogNet(128, 128, len_embedding=512, backbone=args.backbone)

    train(args, recognet, dataset)
