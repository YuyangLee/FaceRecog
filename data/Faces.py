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
    def __init__(self, data_dir, train=False, device='cuda'):
        super().__init__()
        
        self.data_dir = data_dir
        self.device = device
        
        if train:
            self.__getitem__ = self.getitem_train
            self._load_train(data_dir)
        else:
            self.__getitem__ = self.getitem_test
            self._load_test(data_dir)
            
        
    def __len__(self, index):
        pass
        
    def getitem_train(self, index):
        pass
    
    def getitem_test(self, index):
        pass

    def _load_train(self, data_dir):
        pass
    
    def _load_test(self, data_dir):
        pass
