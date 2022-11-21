'''
Author: Aiden Li
Date: 2022-11-11 15:35:31
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-11-11 15:55:17
Description: Dataset for face images.
'''

import json
import os
import uuid
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Faces(Dataset):
    def __init__(self, base_dir, mode='train', device='cuda'):
        super().__init__()
        
        self.device = device
        
        if mode == 'train':
            self.__getitem__ = self.getitem_train
        elif mode == 'valid':
            self.__getitem__ = self.getitem_test
        else:
            self.__getitem__ = self.getitem_test
        self._load_test(base_dir)
            
        
    def __len__(self, index):
        pass
        
    def getitem_train(self, index):
        pass
    
    def getitem_test(self, index):
        pass

    def _load(self, base_dir, mode='train'):
        if mode == 'train' or mode == 'valid':
            self.base_dir = os.path.join(base_dir, "training_set")
            names = json.load(open(os.path.join(self.base_dir, "split.json"), 'r'))[mode]
        elif mode == 'test':
            self.base_dir = os.path.join(base_dir, "test_pair")
            names = os.listdir(os.path.join(self.base_dir, "test_pair"))