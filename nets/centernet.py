import torch
import torch.nn as nn

from .resnet import get_resnet50, Resnet50Decoder, Resnet50Head

class CenternetResnet50(nn.Module):
    
    def __init__(self, n_class, pretrain_path, pretrain = True):
        super(CenternetResnet50, self).__init__()
        self.backbone = get_resnet50(pretrain_path, pretrain)
        self.decoder = Resnet50Decoder()
        self.head = Resnet50Head(n_class=n_class)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.backbone(x) #bsz,2048,16,16
        return self.head(self.decoder(out))