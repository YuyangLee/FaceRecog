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
    def __init__(self,
                base_dir,
                batch_size=16, H=64, W=64,
                mode='train', train_extractor='triplet',
                lazy=False, preload_device='cpu', device='cuda'
                ):
        super().__init__()
        
        self.lazy = lazy
        
        # The amount of positive/negative pairs in a batch. Use batch_size=1 in data_loader
        self.batch_size = batch_size
        
        self.H, self.W = H, W
        self.resizer = Resize((H, W))   # Bilinear
        
        self.preload_device = preload_device
        self.device = device
        
        self.aligned_images_paths = []
        self.rectangles = []
        self.aligned_images = []
        
        if mode == 'train':
            if train_extractor == 'triplet':
                self._getitem = self._train_getitem_triplet
            elif train_extractor == 'pair':
                self._getitem = self._train_getitem_pair
            else:
                raise NotImplementedError("Extractor not implemented.")
            self.train_load(base_dir)
        elif mode == 'valid':
            self._getitem = self.valid_getitem
            self.valid_load(base_dir)
        elif mode == 'test':
            self._getitem = self.test_getitem
        else:
            raise NotImplementedError("Mode not implemented.")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self._getitem(index)
        
    def _train_getitem_triplet(self, index):
        img_anc = self.get_image(index).clone().to(self.device)
        idx_pos, idx_neg = self._get_triplet(index)
        img_pos = self.get_image(idx_pos).clone().to(self.device)
        img_neg = self.get_image(idx_neg).clone().to(self.device)
        label_anc, label_pos, label_neg = torch.tensor(self.idx_to_label[index]).to(self.device), torch.tensor(self.idx_to_label[idx_pos]).to(self.device), torch.tensor(self.idx_to_label[idx_neg]).to(self.device)
        return img_anc, img_pos, img_neg, label_anc, label_pos, label_neg
    
    def _train_getitem_pair(self, index):
        """
        Get item for pair-based methods, returns a positive pair with name index as labels.
        """
        img = self.get_image(index).clone().to(self.device)
        idx_pos = self._get_pos_cp(index)
        img_pos = self.get_image(idx_pos).clone().to(self.device)
        label_0, label_1 = torch.tensor(self.idx_to_label[index]).to(self.device), torch.tensor(self.nameidx_to_label_to_label[idx_pos]).to(self.device)
        return img, img_pos, label_0, label_1
    
    def valid_getitem(self, index):
        img_i, img_j, label = self.all_data[index]
        img_0, img_1 = self.get_image(img_i).clone(), self.get_image(img_j).clone()
        label = torch.tensor(label, device=self.device, dtype=int)
        return img_0, img_1, label
    
    def test_getitem(self, index):
        image_0 = self.get_images(2 * index    ).clone().to(self.device)
        image_1 = self.get_images(2 * index + 1).clone().to(self.device)
        return image_0, image_1
    
    def _get_pos_cp(self, index):
        name = self.idx_to_name[index]
        person_idxs = self.name_to_idx[name]
        pos_idx = random.choice(person_idxs)
        return pos_idx
    
    def _get_triplet(self, index):
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
        rect_x1, rect_y1, rect_x2, rect_y2 = self.rectangles[index][0], self.rectangles[index][1], self.rectangles[index][2], self.rectangles[index][3]
        img = self._crop_and_resize(img, rect_x1, rect_y1, rect_x2, rect_y2)
        return img

    def _load_image(self, aligned_file_path, rectangle, image_idx):
        self.aligned_images_paths.append(aligned_file_path)
        self.rectangles.append(rectangle)
        if not self.lazy:
            self.aligned_images.append(self.train_get_image_lazy(image_idx, self.preload_device))
            self.get_image = self.train_get_image_pre
        else:
            self.get_image = lambda i: self.train_get_image_lazy(i, device=self.device)

    def train_load(self, base_dir):
        self.base_dir = os.path.join(base_dir, "training_set")
        
        self.idx_to_name = []
        self.name_to_idx = defaultdict(list)
        self.name_to_label = {}
        self.idx_to_label = []
        
        meta = json.load(open(os.path.join(self.base_dir, "bb.json"), 'r'))
        names = json.load(open(os.path.join(self.base_dir, "train.json"), 'r'))
        
        meta = { k: v for k, v in meta.items() if k in names }
            
        for i_name, (name, metadata) in enumerate(meta.items()):
            aligned_img_files = metadata['pics_aligned']
            self.name_to_label[name] = i_name
            
            # Prepare lists for data
            for i_file, aligned_img_file in enumerate(aligned_img_files):
                rectangle = metadata['align_params'][aligned_img_file]['rect_aligned']
                rectangle = torch.tensor(rectangle, device=self.device)
                if (rectangle < 0).sum() > 0:
                    print(f"\tSkipped { aligned_img_file } because of error values in rectangle.")
                    continue
                
                self.idx_to_label.append(i_name)
                aligned_file_path = os.path.join(self.base_dir, name, "aligned", aligned_img_file)
                
                self.name_to_idx[name].append(len(self.idx_to_name))
                self.idx_to_name.append(name)
                
                self._load_image(aligned_file_path, rectangle, len(self.idx_to_name) - 1)
                    
        self.len = len(self.idx_to_name)
        tqdm.write(f"Loading { self.len} data instances in training mode...")
        
    def valid_load(self, base_dir):
        self.base_dir = os.path.join(base_dir, "training_set")
        
        self.all_data = []
        
        meta = json.load(open(os.path.join(self.base_dir, "bb.json"), 'r'))
        pairs = json.load(open(os.path.join(self.base_dir, "valid.json"), 'r'))
        
        for pair in pairs:
            data = []
            for i_person, person in enumerate(pair[:-1]):
                img_i = len(self.aligned_images)
                data.append(img_i)
                aligned_img_file = os.path.join(self.base_dir, person[0], "aligned", person[1])
                
                metadata = meta[person[0]]
                if person[1] not in metadata['pics_aligned']:
                    print(f"Skipped { aligned_img_file } because it is not in the metadata.")
                    continue
                rectangle = metadata['align_params'][person[1]]['rect_aligned']
                rectangle = torch.tensor(rectangle, device=self.device)
                
                # Error bbs are excluded when processing pair lists
                self._load_image(aligned_img_file, rectangle, img_i)
                
            data.append(pair[-1])
            self.all_data.append(data)
        self.len = len(self.all_data)
    
        tqdm.write(f"Loading { self.len} data pairs in validation mode...")
        
    def test_load(self, base_dir):
        self.base_dir = os.path.join(base_dir, "test_pair")
        self.len = len(self.idx_to_name) / 2    
    
        self.base_dir = os.path.join(base_dir, "training_set")
        
        self.all_data = []
        
        meta = json.load(open(os.path.join(self.base_dir, "bb.json"), 'r'))
        pairs = json.load(open(os.path.join(self.base_dir, "valid.json"), 'r'))
        
        for pair in pairs:
            data = []
            for i_person, person in enumerate(pair[:-1]):
                img_i = len(self.aligned_images)
                data.append(img_i)
                aligned_img_file = os.path.join(self.base_dir, person[0], "aligned", person[1])
                
                metadata = meta[person[0]]
                if person[1] not in metadata['pics_aligned']:
                    print(f"Skipped { aligned_img_file } because it is not in the metadata.")
                    continue
                rectangle = metadata['align_params'][person[1]]['rect_aligned']
                rectangle = torch.tensor(rectangle, device=self.device)
                
                # Error bbs are excluded when processing pair lists
                self._load_image(aligned_img_file, rectangle, img_i)
                
            data.append(pair[-1])
            self.all_data.append(data)
        self.len = len(self.all_data)
    
        tqdm.write(f"Loading { self.len} data pairs in validation mode...")