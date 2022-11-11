import torch
import torch.nn as nn

class RecogNet(nn.Module):
    def __init__(self, image_h, image_w):
        super(RecogNet, self).__init__()
        self.image_h, self.image_w = image_h, image_w
        
    def forward(self, image):
        pass
        