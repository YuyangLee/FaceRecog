import torch
from torch.utils.data import DataLoader
import argparse

import yaml

with open('config/global.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())