import torch
import torch.nn as nn
import torchvision.models as models
resnet18 = models.resnet18()

class RecogNet(nn.Module):
    def __init__(self, image_h, image_w, len_embedding=256, backbone='resnet18'):
        super(RecogNet, self).__init__()
        self.image_h, self.image_w = image_h, image_w
        
        if backbone == 'resnet18':
            self.image_encoder = models.resnet18(num_classes=len_embedding)
        elif backbone == 'resnet34':
            self.image_encoder = models.resnet34(num_classes=len_embedding)
        elif backbone == 'vgg11_bn':
            self.image_encoder = models.vgg11_bn(num_classes=len_embedding)
        # elif backbone == 'vit_b_16':
        #     self.image_encoder = models.vit_b_16()
        else:
            raise NotImplementedError()
        
        self.dropout = nn.Dropout(0.5)
        self.proj_layer = nn.Linear(len_embedding, len_embedding)
        
    def forward(self, image):
        embd = self.image_encoder(image)
        embd = self.dropout(embd)
        embd = self.proj_layer(embd)
        return embd
        