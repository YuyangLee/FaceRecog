import torch
import torch.nn as nn

from models.ResNet import get_resnet_50, get_resnet_101, get_resnet_152

class RecogNet(nn.Module):
    def __init__(self, image_h, image_w, len_embedding=256, backbone='resnet_101'):
        super(RecogNet, self).__init__()
        self.image_h, self.image_w = image_h, image_w
        
        if backbone == 'resnet_50':
            self.image_encoder = get_resnet_50(len_embedding=len_embedding, channels=3)
        elif backbone == 'resnet_101':
            self.image_encoder = get_resnet_101(len_embedding=len_embedding, channels=3)
        elif backbone == 'resnet_152':
            self.image_encoder = get_resnet_152(len_embedding=len_embedding, channels=3)
        else:
            raise NotImplementedError()
        
    def forward(self, image):
        image = image.permute(0, 3, 1, 2)
        return self.image_encoder(image)
        
        