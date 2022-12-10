'''
Author: Aiden Li
Date: 2022-11-11 15:35:31
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-11-11 15:55:17
Description: Dataset for face images.
'''

from collections import defaultdict
import json
import os
import random
import uuid
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm
import cv2

class Faces(Dataset):
    def __init__(self, base_dir, batch_size=16, H=64, W=64, mode='train', train_ratio=0.8, lazy=False, preload_device='cpu', device='cuda'):
        super().__init__()
        
        self.lazy = lazy
        
        # The amount of positive/negative pairs in a batch. Use batch_size=1 in data_loader
        self.batch_size = batch_size
        
        self.H, self.W = H, W
        self.resizer = Resize((H, W))   # Bilinear
        
        self.preload_device = preload_device
        self.device = device
        
        if mode == 'train':
            self._getitem = self.train_getitem
            self.ratio = train_ratio
        elif mode == 'valid':
            self._getitem = self.train_getitem
            self.ratio = 1 - train_ratio
        elif mode == 'test':
            self._getitem = self.test_getitem
        else:
            raise NotImplementedError("Mode not implemented.")
        self._load(base_dir, mode)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self._getitem(index)
        
    def train_getitem(self, index):
        img_anc = self.get_image(index).clone().to(self.device)
        pos_idx, neg_idx = self.get_triplet(index)
        pos_img = self.get_image(pos_idx).clone().to(self.device)
        neg_img = self.get_image(neg_idx).clone().to(self.device)
        return img_anc, pos_img, neg_img, torch.tensor(index).to(self.device), torch.tensor(pos_idx).to(self.device), torch.tensor(neg_idx).to(self.device)
    
    def test_getitem(self, index):
        image_0 = self.get_images(2 * index    ).clone().to(self.device)
        image_1 = self.get_images(2 * index + 1).clone().to(self.device)
        return image_0, image_1
    
    def get_triplet(self, index):
        name = self.idx_to_name[index]
        person_idxs = self.name_to_idx[name]
        pos_idx = random.choice(person_idxs)
        neg_idx = random.choice(np.arange(0, person_idxs[0]).tolist() + np.arange(person_idxs[-1] + 1, self.len).tolist())
        return pos_idx, neg_idx
            
    def _crop_and_resize(self, img, rect_x1, rect_y1, rect_x2, rect_y2):
        # save_image(torch.tensor(img).permute(2, 0, 1), 'debug/img.png')
        img = img[rect_y1:rect_y2, rect_x1:rect_x2]
        # save_image(torch.tensor(img).permute(2, 0, 1), 'debug/img_cropped.png')
        img = self.resizer(img.permute(2, 0, 1)).permute(1, 2, 0)
        # save_image(torch.tensor(img).permute(2, 0, 1), 'debug/img_resized.png')
        return img
    
    def train_get_image_pre(self, index):
        return self.aligned_images[index].to(self.device)
    
    def train_get_image_lazy(self, index, device='cuda'):
        path = self.aligned_images_paths[index]
        img = torch.tensor(cv2.imread(path, cv2.IMREAD_COLOR)).float().to(device)[:, :, [2, 1, 0]] / 256 # BGR to RGB and scale to [0, 1]
        rect_x1, rect_y1, rect_x2, rect_y2 = self.aligned_image_rect[index][0], self.aligned_image_rect[index][1], self.aligned_image_rect[index][2], self.aligned_image_rect[index][3]
        img = self._crop_and_resize(img, rect_x1, rect_y1, rect_x2, rect_y2)
        return img

    def _load(self, base_dir, mode='train'):
        if mode == 'train' or mode == 'valid':
            self.base_dir = os.path.join(base_dir, "training_set")
            
            self.idx_to_name = []
            self.name_to_idx = defaultdict(list)
            
            self.aligned_images_paths = []
            self.aligned_image_rect = []
            
            self.aligned_images = []
            
        elif mode == 'test':
            self.base_dir = os.path.join(base_dir, "test_pair")
        
        meta = json.load(open(os.path.join(self.base_dir, "bb.json"), 'r'))
        names = list(meta.keys())
        random.shuffle(names)
        names = names[:int(len(names) * self.ratio)]
        
        if 'failure' in names:
            names.remove('failure')
        
        meta = { k: v for k, v in meta.items() if k in names }
            
        print(f"Loading data in { mode } mode...")
        for i_name, (name, metadata) in enumerate(meta.items()):
            aligned_img_files = metadata['pics_aligned']
            
            # Prepare lists for data
            for i_file, aligned_img_file in enumerate(aligned_img_files):
                rectangle = metadata['align_params'][aligned_img_file]['rect_aligned']
                
                rectangle = torch.tensor(rectangle, device=self.device)
                if (rectangle < 0).sum() > 0:
                    print(f"\tSkipped { aligned_img_file } because of error values in rectangle.")
                    continue
                
                aligned_file_path = os.path.join(self.base_dir, name, "aligned", aligned_img_file)
                
                self.name_to_idx[name].append(len(self.idx_to_name))
                self.idx_to_name.append(name)
                self.aligned_images_paths.append(aligned_file_path)
                self.aligned_image_rect.append(rectangle)
                
                if not self.lazy:
                    self.aligned_images.append(self.train_get_image_lazy(len(self.idx_to_name) - 1, self.preload_device))
                    self.get_image = self.train_get_image_pre
                else:
                    self.get_image = lambda i: self.train_get_image_lazy(i, device=self.device)
                    
        self.len = len(self.idx_to_name) / 2 if mode == 'test' else len(self.idx_to_name)
        tqdm.write(f"Loading { self.len} (instances / pairs of) data in { mode } mode...")
        