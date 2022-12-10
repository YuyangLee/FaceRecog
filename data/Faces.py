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
from trochvision.transforms import Resize
from tqdm import tqdm
import cv2

class Faces(Dataset):
    def __init__(self, base_dir, batch_size=64, H=128, W=128, mode='train', lazy=False, preload_device='cpu', device='cuda'):
        super().__init__()
        
        self.lazy = lazy
        
        # The amount of positive/negative pairs in a batch. Use batch_size=1 in data_loader
        self.batch_size = batch_size
        
        self.H, self.W = H, W
        
        self.preload_device = preload_device
        self.device = device
        
        if mode == 'train':
            self.__getitem__ = self.train_getitem
            self.__len__ = lambda: len(self.all_data)
        elif mode == 'valid':
            self.__getitem__ = self.train_getitem
            self.__len__ = lambda: len(self.all_data)
        else:
            self.__getitem__ = self.test_getitem
            self.__len__ = lambda: len(self.idx_to_name)
        self._load_test(base_dir)
        
    @torch.no_grad
    def train_getitem(self, index):
        image_0 = self.get_image(index).clone().to(self.devicee)
        triplet, labels = self.get_triplet(index)
        triplet_images = [ self.get_image(j).clone().to(self.devicee) for j in triplet ]
        triplet_images = torch.stack(triplet_images, dim=0)
        
        return image_0, triplet_images, labels
    
    @torch.no_grad
    def test_getitem(self, index):
        image_0 = self.get_images(2 * index    ).clone().to(self.devicee)
        image_1 = self.get_images(2 * index + 1).clone().to(self.devicee)
        return image_0, image_1
    
    def get_triplet(self, index):
        # TODO: Get amount of positive and negative pairs: n_p = min(B, N), n_n = B - n_p
        # TODO: Sample positive pairs.
        # TODO: Chooses collided pair and random pairs to flip
        # TODO: Return index and labels
        pass
    
    def train_get_image_pre(self, index):
        i_name, i_file = self.all_data[index]
        return self.aligned_images[i_name][i_file]
    
    def train_get_image_lazy(self, index):
        i_name, i_file = self.all_data[index]
        path = self.aligned_images_paths[i_name][i_file]
        img = torch.tensor(cv2.imread(path, cv2.IMREAD_COLOR)).float().to(self.preload_device)
        rect_x1, rect_y1, rect_x2, rect_y2 = self.aligned_image_rect[i_name][1], self.aligned_image_rect[i_name][3], self.aligned_image_rect[i_name][0], self.aligned_image_rect[i_name][2]
        # TODO: Pad the image for indicing
        # TODO: Compute the expanded rectangle
        # TODO: Crop with the expanded rectangle
        # TODO: Resize image to H x W
        return img

    @torch.no_grad
    def _load(self, base_dir, mode='train'):
        if mode == 'train' or mode == 'valid':
            self.base_dir = os.path.join(base_dir, "training_set")
            names = json.load(open(os.path.join(self.base_dir, "split.json"), 'r'))[mode]
            
            self.idx_to_name = []
            self.name_to_idx = {}
            
            self.aligned_images_paths = []
            self.aligned_image_rect = []
            
            self.aligned_images = []
            
            self.all_data = []
            
        elif mode == 'test':
            self.base_dir = os.path.join(base_dir, "test_pair")
            names = os.listdir(os.path.join(self.base_dir, "test_pair"))
            
        tqdm.write("Loading data...")
        for i_name, (name, metadata) in tqdm(enumerate(names.items())):
            aligned_img_files = metadata['pics_aligned']
            
            # Prepare lists for data
            self.aligned_images_paths.append([])
            self.aligned_image_rect.append([])
            self.aligned_images.append([])
            
            for i_file, aligned_img_file in enumerate(aligned_img_files):
                self.all_data.append((i_name, i_file))
                
                rectangle = metadata['align_params'][aligned_img_file]['rect_aligned']
                aligned_file_path = os.path.join(self.base_dir, name, "aligned", aligned_img_file)
                
                self.name_to_idx[name] = len(self.idx_to_name)
                self.idx_to_name.append(name)
                self.aligned_images_paths[-1].append(aligned_file_path)
                self.aligned_image_rect.append(torch.array(rectangle))
                
                if self.lazy:
                    self.aligned_images[-1].append(self.train_get_image_lazy(len(self.all_data) - 1))
                
                self.get_image = self.train_get_image_lazy if self.lazy else self.train_get_image_pre
            