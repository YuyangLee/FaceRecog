import torch
from torch.utils.data import DataLoader
from data.Faces import Faces
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=100, type=int)

parser.add_argument("--train_ratio", default=0.7, type=float)

parser.add_argument("--backbone", default="resnet_50", type=str)
parser.add_argument("--loss", default="triplet", type=str)

args = parser.get_args()

def train(args):

