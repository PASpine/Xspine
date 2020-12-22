import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

# support route shortcut
class MatrixNet(nn.Module):
    def __init__(self, num_classes, in_channels, num_layers_stage=[1, 2, 8, 8, 4]):
        super(MatrixNet, self).__init__()

        # Initial parameters
        self.Nchannels = 32
        self.seen = 0
        self.num_classes = num_classes
        self.num_anchors = 3
        self.width = 512
        self.height = 512
        out_channels = (5+self.num_classes) * self.num_anchors

        # Initial convolution layers
        self.conv1_1 = ConvLayer_GN(in_channels, self.Nchannels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ConvLayer_GN(self.Nchannels, self.Nchannels * 2, kernel_size=3, stride=2, padding=1)
        self.C1 = self._make_layer(num_layers_stage[0], self.Nchannels * 2)

        self.conv2 = ConvLayer_GN(self.Nchannels * 2, self.Nchannels * 4, kernel_size=3, stride=2, padding=1)
        self.C2 = self._make_layer(num_layers_stage[1], self.Nchannels * 4)

        self.conv3 = ConvLayer_GN(self.Nchannels * 4, self.Nchannels * 8, kernel_size=3, stride=2, padding=1)
        self.conv3_up = ConvLayer_GN(self.Nchannels * 8, self.Nchannels * 8, kernel_size=3, stride=(1,2), padding=1)
        self.conv3_down = ConvLayer_GN(self.Nchannels * 8, self.Nchannels * 8, kernel_size=3, stride=(2, 1), padding=1)
        self.C3 = self._make_layer(num_layers_stage[2], self.Nchannels * 8)

        self.conv4 = ConvLayer_GN(self.Nchannels * 8, self.Nchannels * 16, kernel_size=3, stride=2, padding=1)
        self.conv4_up = ConvLayer_GN(self.Nchannels * 16, self.Nchannels * 16, kernel_size=3, stride=(1,2), padding=1)
        self.conv4_down = ConvLayer_GN(self.Nchannels * 16, self.Nchannels * 16, kernel_size=3, stride=(2, 1), padding=1)
        self.C4 = self._make_layer(num_layers_stage[3], self.Nchannels * 16)

        self.conv5 = ConvLayer_GN(self.Nchannels * 16, self.Nchannels * 32, kernel_size=3, stride=2, padding=1)
        self.C5 = self._make_layer(num_layers_stage[4], self.Nchannels * 32)

        # self.conv = ConvLayer_GN(channels, int(channels / 2), kernel_size=1, stride=1, padding=0)
        self.detect1 = DetectBlock(self.Nchannels * 32, self.Nchannels * 32, out_channels)

        # self.upsample1 = UpsampleBlock(self.Nchannels * 16)
        self.detect2 = DetectBlock(self.Nchannels * 16, self.Nchannels * (16+8+8+8), out_channels)
        self.upconv2 = ConvLayer_GN(self.Nchannels * 16, self.Nchannels * 8, kernel_size=1, stride=1, padding=0)

        # self.upsample2 = UpsampleBlock(self.Nchannels * 8)
        self.upconv3 = ConvLayer_GN(self.Nchannels * 8, self.Nchannels * 4, kernel_size=1, stride=1, padding=0)
        self.detect3 = DetectBlock(self.Nchannels * 8, self.Nchannels * (8 + 4 +4 +4), out_channels)

    def _make_layer(self, num_blocks, planes):
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x) # 1 32 416 416
        x = self.conv1_2(x) # 1 64 208 208
        x = self.C1(x) # 1 64 208 208

        x = self.conv2(x) # 1 128 104 104
        x = self.C2(x)

        x3 = self.conv3(x) # 1 256 52 52
        x3_up = self.conv3_up(self.conv3_up(x3)) # 1, 512 ,26, 13
        x3_down = self.conv3_up(self.conv3_down(x3)) # 1 512 13 26
        c3 = self.C3(x3) # 1 256 52 52
        c3_up = self.C3(x3_up) # 1, 512 ,26, 13
        c3_down = self.C3(x3_down) # 1 512 13 26

        x4 = self.conv4(c3) # 1 512 26 26
        x4_up = self.conv4_up(x4) # 1, 512 ,26, 13
        x4_down = self.conv4_down(x4) # 1 512 13 26
        c4 = self.C4(x4) # 1 512 26 26
        c4_up = self.C4(x4_up) # 1, 512 ,26, 13
        c4_down = self.C4(x4_down) # 1 512 13 26

        x = self.conv5(c4) # 1 1024 13 13
        c5 = self.C5(x) # 1 1024 13 13

        output1, x = self.detect1(c5) # 1 27 13 13   1 512 13 13

        x = self.upconv2(x) # 1 256 13 13
        c4_up = self.upconv2(c4_up)
        c4_down = self.upconv2(c4_down)
        x = F.interpolate(x, (c4.shape[-2],c4.shape[-1]), mode='bilinear', align_corners=True) #  1 256 26 26
        c4_up = F.interpolate(c4_up, (c4.shape[-2], c4.shape[-1]), mode='bilinear', align_corners=True)
        c4_down = F.interpolate(c4_down, (c4.shape[-2], c4.shape[-1]), mode='bilinear', align_corners=True)
        x = torch.cat((x, c4, c4_up, c4_down), 1) # 1, 1280, 26, 26
        output2, x = self.detect2(x) #  1 27 26 26  1 256 26 26

        x = self.upconv3(x) # 1 128 26 26
        c3_up = self.upconv3(c3_up)
        c3_down = self.upconv3(c3_down)
        x = F.interpolate(x, (c3.shape[-2],c3.shape[-1]), mode='bilinear', align_corners=True) #  1 256 26 26
        c3_up = F.interpolate(c3_up, (c3.shape[-2], c3.shape[-1]), mode='bilinear', align_corners=True)
        c3_down = F.interpolate(c3_down, (c3.shape[-2], c3.shape[-1]), mode='bilinear', align_corners=True)
        x = torch.cat((x, c3, c3_up, c3_down), 1) # 1, 640, 26, 26
        # x = self.upsample2(x, c3) # 1 384 52 52
        output3, x = self.detect3(x)#  1,27,52,52    1 128 52 52

        return [output1, output2, output3] #, c5, c4, c3

    def load_pretrained_weights(self, weightfile):
        pretrained_params = torch.load(weightfile, map_location=torch.device('cpu'))
        model_dict = self.state_dict()
        pretrained_params = {k: v for k, v in pretrained_params.items() if k in model_dict}
        self.seen = 0
        model_dict.update(pretrained_params)
        self.load_state_dict(model_dict)
        print('Load Pretrained Weights from %s... Done!!!' % weightfile)
        del pretrained_params
        del model_dict

    def load_weights(self, weightfile):
        params = torch.load(weightfile, map_location=torch.device('cpu'))
        if 'seen' in params.keys():
            self.seen = params['seen']
            del params['seen']
        else:
            self.seen = 0
        self.load_state_dict(params)
        print('Load Weights from %s... Done!!!' % weightfile)
        del params

    def save_weights(self, outfile):
        params = self.state_dict()
        params['seen'] = self.seen
        torch.save(params, outfile)
        del params

    def load_binary_weights(self, weightfile):
        pretrained_params = torch.load(weightfile)
        model_dict = self.state_dict()

        key1 = list(pretrained_params.keys())
        key2 = list(model_dict.keys())
        j=0
        for i in range(len(key2)):
            k_1 = key1[j].split('.')
            k_2 = key2[i].split('.')
            if(k_1[-1] == k_2[-1]):
                pretrained_params[key2[i]] = pretrained_params.pop(key1[j])
                j += 1
        pretrained_params = {k: v for k, v in pretrained_params.items() if k in model_dict}
        self.seen = 0
        model_dict.update(pretrained_params)
        self.load_state_dict(model_dict)
        del pretrained_params
        del model_dict

class ConvLayer_GN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, groups=32, deconv=False):
        super(ConvLayer_GN, self).__init__()
        if deconv:
            self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.norm(self.conv2d(x)))
        return y

class ShortcutBlock(nn.Module):
    def __init__(self, channels, in_channels=None, shortcut=True, groups=32):
        super(ShortcutBlock, self).__init__()

        self.shortcut =shortcut
        if in_channels is not None:
            self.conv1 = nn.Conv2d(in_channels, channels // 2, kernel_size=1, stride=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, route=False):
        if self.shortcut:
            y = self.leakyrelu(self.norm1(self.conv1(x)))
            y = self.leakyrelu(self.norm2(self.conv2(y)))
            y = y + x
            return y
        elif route:
            y1 = self.leakyrelu(self.norm1(self.conv1(x)))
            y = self.leakyrelu(self.norm2(self.conv2(y1)))
            return y, y1
        else:
            y = self.leakyrelu(self.norm1(self.conv1(x)))
            y = self.leakyrelu(self.norm2(self.conv2(y)))
            return y

class DetectBlock(nn.Module):
    def __init__(self, channels, in_channels, out_channels, shortcut=False):

        super(DetectBlock, self).__init__()
        self.block1 = ShortcutBlock(channels, in_channels, shortcut=False)
        self.block2 = ShortcutBlock(channels, shortcut=shortcut)
        self.block3 = ShortcutBlock(channels, shortcut=False)
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y, y1 = self.block3(y, True)
        y = self.conv(y)
        return y, y1

class UpsampleBlock(nn.Module):
    def __init__(self, channels):

        super(UpsampleBlock, self).__init__()
        self.conv = ConvLayer_GN(channels, int(channels / 2), kernel_size=1, stride=1, padding=0)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, x1):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, x1), 1)
        return x

class FPN(nn.Module):
    def __init__(self, Nchannels=32, mask_branch_channels=256):
        super(FPN, self).__init__()

        self.C3_tail = ConvLayer_GN(Nchannels * 8, mask_branch_channels, 1, 1)
        self.C4_tail = ConvLayer_GN(Nchannels * 16, mask_branch_channels, 1, 1)
        self.C5_tail = ConvLayer_GN(Nchannels * 32, mask_branch_channels, 1, 1)

        self.C3_conv = ConvLayer_GN(mask_branch_channels, mask_branch_channels, 3, 1, padding=1)
        self.C4_conv = ConvLayer_GN(mask_branch_channels, mask_branch_channels, 3, 1, padding=1)
        self.C5_conv = ConvLayer_GN(mask_branch_channels, mask_branch_channels, 3, 1, padding=1)

    def forward(self, x):
        p5 = self.C5_tail(x[0])
        p4 = self.C4_tail(x[1]) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=True)
        p3 = self.C3_tail(x[2]) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)

        p5 = self.C5_conv(p5)
        p4 = self.C4_conv(p4)
        p3 = self.C3_conv(p3)

        return [p5, p4, p3]

class Mask_Branch(nn.Module):
    def __init__(self, in_channels, num_classes, align_size):
        super(Mask_Branch, self).__init__()

        self.align_size = align_size
        self.block1 = ShortcutBlock(in_channels)
        self.deconv1 = ConvLayer_GN(in_channels, in_channels, 2, 2, deconv=True)
        self.block2 = ShortcutBlock(in_channels)
        self.deconv2 = ConvLayer_GN(in_channels, in_channels, 2, 2, deconv=True)
        self.block3 = ShortcutBlock(in_channels)

        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x, rois):
        x = RoI_Align(x, rois, self.align_size)
        x = self.block1(x)
        x = self.deconv1(x)
        x = self.block2(x)
        x = self.deconv2(x)
        x = self.block3(x)
        x = self.conv(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1,1,416,416)
    model = MatrixNet(4,1)
    out = model.forward(x)