import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# support route shortcut
class YOLO(nn.Module):
    def __init__(self, num_classes, in_channels, num_anchors=3):
        super(YOLO, self).__init__()

        # Initial parameters
        self.Nchannels = 32
        self.seen = 0
        self.width = 448
        self.height = 800
        self.num_anchors = num_anchors
        out_channels = (6 + num_classes) * self.num_anchors

        # Initial convolution layers
        self.conv0 = ConvLayer_BN(in_channels, self.Nchannels, 3, 1, 1)

        self.down1 = downBlock(self.Nchannels, self.Nchannels * 2, 3, 2, 1)
        self.down2 = downBlock(self.Nchannels * 2, self.Nchannels * 4, 3, 2, 1)
        self.down3 = downBlock(self.Nchannels * 4, self.Nchannels * 8, 3, 2, 1, num_blocks=8)
        self.down4 = downBlock(self.Nchannels * 8, self.Nchannels * 16, 3, 2, 1, num_blocks=8)
        self.down5 = downBlock(self.Nchannels * 16, self.Nchannels * 32, 3, 2, 1, num_blocks=4)

        self.detect5 = DetectBlock(self.Nchannels * 32, self.Nchannels * 32, out_channels)

        self.up4 = UpsampleBlock(self.Nchannels * 16)
        self.detect4 = DetectBlock(self.Nchannels * 16, self.Nchannels * (16 + 8), out_channels)

        self.up3 = UpsampleBlock(self.Nchannels * 8)
        self.detect3 = DetectBlock(self.Nchannels * 8, self.Nchannels * (8 + 4), out_channels)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        output5, x = self.detect5(x5)

        x = self.up4(x, x4)
        output4, x = self.detect4(x)

        x = self.up3(x, x3)
        output3, x = self.detect3(x)

        return [output5, output4, output3]

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_model(self):
        for param in self.parameters():
            param.requires_grad = True

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


class ConvLayer_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvLayer_BN, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv2d(x)))
        return y

class ShortcutBlock(nn.Module):
    def __init__(self, channels):
        super(ShortcutBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn1(self.conv1(x)))
        y = self.leakyrelu(self.bn2(self.conv2(y)))
        y = y + x
        return y

class downBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_blocks=2):
        super(downBlock, self).__init__()

        self.down1 = ConvLayer_BN(in_channels, out_channels, kernel_size, stride, padding)
        layers = []
        for i in range(num_blocks):
            layers.append(ShortcutBlock(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(self.down1(x))

class UpsampleBlock(nn.Module):
    def __init__(self, channels):

        super(UpsampleBlock, self).__init__()
        self.conv = ConvLayer_BN(channels, channels // 2, kernel_size=1, stride=1)

    def forward(self, x, x_down):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, x_down), 1)
        return x

class DetectBlock(nn.Module):
    def __init__(self, channels, in_channels, out_num_classes):

        super(DetectBlock, self).__init__()
        self.layer1 = ConvLayer_BN(in_channels, channels // 2, 1, 1)
        self.layer2 = ConvLayer_BN(channels // 2, channels, 3, 1, 1)
        self.layer3 = ConvLayer_BN(channels, channels// 2, 1, 1)
        self.layer4 = ConvLayer_BN(channels // 2, channels, 3, 1, 1)
        self.layer5 = ConvLayer_BN(channels, channels // 2, 1, 1)
        self.layer6 = ConvLayer_BN(channels // 2, channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(channels, out_num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        y_feature = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        y_out = self.conv_out(self.layer6(y_feature))
        return y_out, y_feature


# def test():
#     output = model(data)
#     Outputs = [output[0], output[1], output[2]]
#
#     loss, nGT, nCorrect, nProposals = YoloLoss(Outputs, target, yolo_config.num_classes, anchors, 2,
#                                                processed_batches * batch_size, device)