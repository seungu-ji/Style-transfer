# Universal Style Transfer via Feature Transforms

import torch
import torch.nn as nn


## Whitening and Coloring Transform
class WCT(nn.Module):
    def __init__(self, paths):
        super(WCT, self).__init__()
        encoder_path1, encoder_path2, encoder_path3, encoder_path4, encoder_path5, decoder_path1, decoder_path2, decoder_path3, decoder_path4, decoder_path5 = paths

        encoder1_param = torch.load(encoder_path1)
        encoder2_param = torch.load(encoder_path2)        
        encoder3_param = torch.load(encoder_path3)
        encoder4_param = torch.load(encoder_path4)        
        encoder5_param = torch.load(encoder_path5)
        decoder1_param = torch.load(decoder_path1)
        decoder2_param = torch.load(decoder_path2)
        decoder3_param = torch.load(decoder_path3)
        decoder4_param = torch.load(decoder_path4)
        decoder5_param = torch.load(decoder_path5)

    def whiten_and_color(self, c, s):
        pass


# ReflectionPad2d - Load weight for Conv2d - ReLU
class load_conv(nn.Module):
    def __init__(self, vgg, num_param, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reflect=True, relu=True):
        super().__init__()
        
        layers = []

        if reflect:
            layers += [nn.ReflectionPad2d((1,1,1,1))]

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(num_param)].float())
        self.conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(num_param)].float())
        layers += [self.conv]

        if relu:
            layers += nn.ReLU(inplace=True)
        
        self.layers = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.layers(x)


class encoder1(nn.Module):
    def __init__(self, vgg):
        super(encoder1, self).__init__()
        self.conv1 = load_conv(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = load_conv(vgg, 2, 3, 64, 3, 1, 0, True, True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class encoder2(nn.Module):
    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.conv1 = load_conv(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = load_conv(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = load_conv(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv4 = load_conv(vgg, 9, 64, 128, 3, 1, 0, True, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)

        x = self.conv4(x)

        return x

class encoder3(nn.Module):
    def __init__(self, vgg):
        super(encoder3, self).__init__()
        self.conv1 = load_conv(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = load_conv(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = load_conv(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = load_conv(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = load_conv(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = load_conv(vgg, 16, 128, 256, 3, 1, 0, True, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.maxpool2(x)

        x = self.conv6(x)

        return x

class encoder4(nn.Module):
    def __init__(self, vgg):
        super(encoder4, self).__init__()
        self.conv1 = load_conv(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = load_conv(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = load_conv(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = load_conv(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = load_conv(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = load_conv(vgg, 16, 128, 256, 3, 1, 0, True, True)

        self.conv7 = load_conv(vgg, 19, 256, 256, 3, 1, 0, True, True)

        self.conv8 = load_conv(vgg, 22, 256, 256, 3, 1, 0, True, True)
        
        self.conv9 = load_conv(vgg, 25, 256, 256, 3, 1, 0, True, True)

        # 28 x 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv10 = load_conv(vgg, 29, 256, 512, 3, 1, 0, True, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.maxpool2(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.maxpool3(x)
        
        x = self.conv10(x)
        
        return x

class encoder5(nn.Module):
    def __init__(self, vgg):
        super(encoder5, self).__init__()
        self.conv1 = load_conv(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = load_conv(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = load_conv(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = load_conv(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = load_conv(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = load_conv(vgg, 16, 128, 256, 3, 1, 0, True, True)

        self.conv7 = load_conv(vgg, 19, 256, 256, 3, 1, 0, True, True)

        self.conv8 = load_conv(vgg, 22, 256, 256, 3, 1, 0, True, True)
        
        self.conv9 = load_conv(vgg, 25, 256, 256, 3, 1, 0, True, True)

        # 28 x 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv10 = load_conv(vgg, 29, 256, 512, 3, 1, 0, True, True)

        self.conv11 = load_conv(vgg, 32, 512, 512, 3, 1, 0, True, True)

        self.conv12 = load_conv(vgg, 35, 512, 512, 3, 1, 0, True, True)

        self.conv13 = load_conv(vgg, 38, 512, 512, 3, 1, 0, True, True)

        # 14 x 14
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv14 = load_conv(vgg, 42, 512, 512, 3, 1, 0, True, True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.maxpool2(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.maxpool3(x)
        
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.maxpool4(x)

        x = self.conv14(x)

        return x


class decoder1(nn.Module):
    def __init__(self, dvgg):
        super(decoder1, self).__init__()
        self.conv3 = load_conv(dvgg, 1, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv3(x)

        return x

class decoder2(nn.Module):
    def __init__(self, dvgg):
        super(decoder2, self).__init__()
        # 112 x 112
        self.conv5 = load_conv(dvgg, 1, 128, 64, 3, 1, 0, True, True)
        
        # 224 x 224
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv6 = load_conv(dvgg, 5, 64, 64, 3, 1, 0, True, True)

        self.conv7 = load_conv(dvgg, 8, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv5(x)

        x = self.unpool1(x)

        x = self.conv6(x)
        x = self.conv7(x)

        return x

class decoder3(nn.Module):
    def __init__(self, dvgg):
        super(decoder3, self).__init__()
        # 56 x 56
        self.conv7 = load_conv(dvgg, 1, 256, 128, 3, 1, 0, True, True)

        # 112 x 112
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv8 = load_conv(dvgg, 5, 128, 128, 3, 1, 0, True, True)