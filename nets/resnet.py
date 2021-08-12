import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_resnet50(pretrain_path, pretrain=True):
    model = resnet50(False)
    if pretrain:
        state_dict = torch.load(pretrain_path)
        model.load_state_dict(state_dict)

    features = list(
        [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features = nn.Sequential(*features)
    return features

class Resnet50Decoder(nn.Module):
    '''
    2048x16x16->64,128,128
    '''
    def __init__(self, inplanes=2048):
        super().__init__()
        self.inplanes = inplanes
        self.model = self._make_deconv_layer(3, [256, 128, 64], [4,4,4])
    def _make_deconv_layer(self, n_layer, filter_list, kernel_size_list):

        layer_list = []
        for i in range(n_layer):
            n_filter = filter_list[i]

            layer_i = nn.Sequential(
                         nn.ConvTranspose2d(self.inplanes,
                               n_filter,
                               kernel_size_list[i],
                               stride=2,
                               padding=1,
                               bias=False ),
                          nn.BatchNorm2d(n_filter),
                          nn.ReLU()
                      )
            layer_list.append(layer_i)
            self.inplanes = n_filter

        return nn.Sequential(*layer_list)
    def forward(self, x):
        # print('from decoder forward: x.shape', x.shape)
        return self.model(x)

class Resnet50Head(nn.Module):

    def __init__(self, inplanes=64, channel=64, n_class=20):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(inplanes, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, n_class, kernel_size=1, stride=1, padding=0)
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(inplanes, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(inplanes, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        cls = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)

        return cls, wh, offset
