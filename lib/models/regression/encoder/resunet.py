'''ResUNet in PyTorch.
https://github.com/qianqianwang68/caps/blob/master/CAPS/network.py
Reference:
[1] Zhengxin Zhang, Qingjie Liu
    Road Extraction by Deep Residual U-Net. arXiv:1711.10684
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.regression.encoder.preact import PreActBlock, PreActBottleneck


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale,
                                      mode='bilinear', align_corners=True)
        return self.conv1(x)


class ResUNet(nn.Module):
    def __init__(self, cfgmodel, num_in_layers=3):
        super().__init__()
        filters = [256, 512, 1024, 2048]
        self.in_planes = 64
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(
                num_in_layers, 64, kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False)
        else:
            self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # H/2
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4
        # encoder
        block_type = [PreActBlock, PreActBottleneck]
        block = block_type[cfgmodel.BLOCK_TYPE]
        num_blocks = [int(x) for x in cfgmodel.NUM_BLOCKS.strip().split("-")]
        self.encoder1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # H/4
        self.encoder2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # H/8
        self.encoder3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # H/16

        # decoder
        self.not_concat = getattr(cfgmodel, "NOT_CONCAT", False)
        self.upconv4 = upconv(filters[2], 512, 3, 2)
        if not self.not_concat:
            self.iconv4 = conv(filters[1] + 512, 512, 3, 1)
        else:
            self.iconv4 = conv(512, 512, 3, 1)

        self.upconv3 = upconv(512, 256, 3, 2)
        if not self.not_concat:
            self.iconv3 = conv(filters[0] + 256, 256, 3, 1)
        else:
            self.iconv3 = conv(256, 256, 3, 1)

        num_out_layers = getattr(cfgmodel, "NUM_OUT_LAYERS", 128)
        self.num_out_layers = num_out_layers
        self.outconv = conv(256, num_out_layers, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # encoding
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # decoding
        x = self.upconv4(x4)
        if not self.not_concat:
            x = self.skipconnect(x3, x)
        x = self.iconv4(x)

        x = self.upconv3(x)
        if not self.not_concat:
            x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.outconv(x)
        return x
