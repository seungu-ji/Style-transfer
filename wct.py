# Universal Style Transfer via Feature Transforms

import torch
import torch.nn as nn

"""
ARCHITECTURE for ENCODERS

Tuple: Convolution2d => (in_channels, out_channels, kernel_size, stride, padding)
List: ReflectionPad - Convolution2d Tuple - weight index - ReLU
"encoder {n} end": index n encoder is end
"M": MaxPool2d
"""
encoder_architecture = [
    ## Encoder1: conv1 - conv2
    [False, (3,3,1,1,0), 0, False], [True, (3,64,3,1,0), 2, True], "encoder 1 end",
    ## Encoder2: Encoder1 - conv3 - maxpool - conv4
    [True, (64,64,3,1,0), 5, True], "M", [True, (64,128,3,1,0), 9, True], "encoder 2 end",
    ## Encoder3: Encoder2 - conv5 - maxpool - conv6 
    [True, (128,128,3,1,0), 12, True], "M", [True, (128,256,3,1,0), 16, True], "encoder 3 end",
    ## Encoder4: Encoder3 - conv7 - conv8 - conv9 - maxpool - conv10
    [True, (256,256,3,1,0), 19, True], [True, (256,256,3,1,0), 22, True], [True, (256,256,3,1,0), 25, True],
    "M", [True, (256,512,3,1,0), 29, True], "encoder 4 end",
    ## Encoder5: Encoder4 - conv11 - conv12 - conv13 - maxpool - conv14
    [True, (512,512,3,1,0), 32, True], [True, (512,512,3,1,0), 35, True], [True, (512,512,3,1,0), 38, True],
    "M", [True, (512,512,3,1,0), 42, True], "encoder 5 end"
]


def wct_style_transfer(wct, alpha, content_image, style_image, csf):
    sf5 = wct.e5(style_image)
    cf5 = wct.e5(content_image)
    sf5 = sf5.data.cpu().squeeze(0)
    cf5 = cf5.data.cpu().squeeze(0)
    csf5 = wct.transform(cf5,sf5,csf,alpha)
    im5 = wct.d5(csf5)

    sf4 = wct.e4(style_image)
    cf4 = wct.e4(im5)
    sf4 = sf4.data.cpu().squeeze(0)
    cf4 = cf4.data.cpu().squeeze(0)
    csf4 = wct.transform(cf4,sf4,csf,alpha)
    im4 = wct.d4(csf4)

    sf3 = wct.e3(style_image)
    cf3 = wct.e3(im4)
    sf3 = sf3.data.cpu().squeeze(0)
    cf3 = cf3.data.cpu().squeeze(0)
    csf3 = wct.transform(cf3,sf3,csf,alpha)
    im3 = wct.d3(csf3)

    sf2 = wct.e2(style_image)
    cf2 = wct.e2(im3)
    sf2 = sf2.data.cpu().squeeze(0)
    cf2 = cf2.data.cpu().squeeze(0)
    csf2 = wct.transform(cf2,sf2,csf,alpha)
    im2 = wct.d2(csf2)

    sf1 = wct.e1(style_image)
    cf1 = wct.e1(im2)
    sf1 = sf1.data.cpu().squeeze(0)
    cf1 = cf1.data.cpu().squeeze(0)
    csf1 = wct.transform(cf1,sf1,csf,alpha)
    im1 = wct.d1(csf1)

    #torchvision.utils.save_image(im1.data.cpu().float(), 'output.jpg')

    return im1.data.cpu().float()

## Whitening and Coloring Transform
class WCT(nn.Module):
    def __init__(self, paths):
        super(WCT, self).__init__()
        encoder_path1, encoder_path2, encoder_path3, encoder_path4, encoder_path5, decoder_path1, decoder_path2, decoder_path3, decoder_path4, decoder_path5 = paths

        ## Load pre-trained network
        # load encoder
        encoder1_param = torch.load(encoder_path1)
        encoder2_param = torch.load(encoder_path2)
        encoder3_param = torch.load(encoder_path3)
        encoder4_param = torch.load(encoder_path4)
        encoder5_param = torch.load(encoder_path5)
        # load decoder
        decoder1_param = torch.load(decoder_path1)
        decoder2_param = torch.load(decoder_path2)
        decoder3_param = torch.load(decoder_path3)
        decoder4_param = torch.load(decoder_path4)
        decoder5_param = torch.load(decoder_path5)

        self.e1 = encoder1(encoder1_param)
        self.d1 = decoder1(decoder1_param)
        self.e2 = encoder2(encoder2_param)
        self.d2 = decoder2(decoder2_param)
        self.e3 = encoder3(encoder3_param)
        self.d3 = decoder3(decoder3_param)
        self.e4 = encoder4(encoder4_param)
        self.d4 = decoder4(decoder4_param)
        self.e5 = encoder5(encoder5_param)
        self.d5 = decoder5(decoder5_param)


    # content feature, style feature
    def whiten_and_color(self, cf, sf):
        # content size
        c_size = cf.size()
        # channels x (height x width)
        # content mean
        c_mean = torch.mean(cf,1)
        c_mean = c_mean.unsqueeze(1).expand_as(cf)
        cf = cf - c_mean
        
        # content conv
        c_conv = torch.mm(cf, cf.t()).div(c_size[1]-1) + torch.eye(c_size[0]).double()
        #c_u, c_e, c_v = torch.linalg.svd(c_conv,some=False)
        c_u, c_e, c_v = torch.svd(c_conv, some=False)

        k_c = c_size[0]
        for i in range(c_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break
        

        # style size
        s_size = sf.size()
        # style mean
        s_mean = torch.mean(sf,1)
        sf = sf - s_mean.unsqueeze(1).expand_as(sf)

        # style conv
        s_conv = torch.mm(sf, sf.t()).div(s_size[1]-1)
        s_u, s_e, s_v = torch.svd(s_conv, some=False) #torch.linalg.svd(s_conv, some=False)

        k_s = s_size[0]
        for i in range(s_size[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break


        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:,0:k_c].t()))
        whiten_cf = torch.mm(step2, cf)

        s_d = (s_e[0:k_s]).pow(0.5)
        target = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s], torch.diag(s_d)), (s_v[:,0:k_s].t())), whiten_cf)
        target = target + s_mean.unsqueeze(1).expand_as(target)

        return target
        
    def transform(self, cf, sf, csf, alpha):
        cf = cf.double()
        sf = sf.double()

        cc, cw, ch = cf.size(0), cf.size(1), cf.size(2)
        _, sw, sh = sf.size(0), sf.size(1), sf.size(2)
        cfview = cf.view(cc, -1)
        sfview = sf.view(cc, -1)

        target = self.whiten_and_color(cfview, sfview)
        target = target.view_as(cf)
        ccsf = alpha *  target + (1.0 - alpha) * cf
        ccsf = ccsf.float().unsqueeze(0)

        with torch.no_grad():
            csf.resize_(ccsf.size()).copy_(ccsf)

        return csf

## ReflectionPad2d - Load weight for Conv2d - ReLU
class RCR2d(nn.Module):
    def __init__(self, vgg, num_param, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reflect=True, relu=True):
        super().__init__()
        
        layers = []

        if reflect:
            layers += [nn.ReflectionPad2d((1,1,1,1))]

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(num_param)].float())
        conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(num_param)].float())
        layers += [conv]

        if relu:
            layers += [nn.ReLU(inplace=True)]
        
        self.layers = nn.Sequential(*layers)

        
    def forward(self, x):
        return self.layers(x)


## Encoder
class encoder1(nn.Module):
    def __init__(self, vgg):
        super(encoder1, self).__init__()
        self.architecture = encoder_architecture
        self.network = self._create_conv_layers(self.architecture, vgg)

    def _create_conv_layers(self, architecture, vgg):
        layers = []

        for x in architecture:
            if type(x) == list:
                reflec = x[0]
                conv_layers = x[1]
                weight_index = x[2]
                activation = x[3]

                if reflec:
                    layers += [nn.ReflectionPad2d((1,1,1,1))]
                
                conv = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=conv_layers[2], stride=conv_layers[3], padding=conv_layers[4])
                conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(weight_index)].float())
                conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(weight_index)].float())
                layers += [conv]

                if activation:
                    layers += [nn.ReLU(inplace=True)]

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                

            elif x == "encoder 1 end":
                break

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class encoder2(nn.Module):
    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.architecture = encoder_architecture
        self.network = self._create_conv_layers(self.architecture, vgg)

    def _create_conv_layers(self, architecture, vgg):
        layers = []

        for x in architecture:
            if type(x) == list:
                reflec = x[0]
                conv_layers = x[1]
                weight_index = x[2]
                activation = x[3]

                if reflec:
                    layers += [nn.ReflectionPad2d((1,1,1,1))]
                
                conv = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=conv_layers[2], stride=conv_layers[3], padding=conv_layers[4])
                conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(weight_index)].float())
                conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(weight_index)].float())
                layers += [conv]

                if activation:
                    layers += [nn.ReLU(inplace=True)]

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                

            elif x == "encoder 2 end":
                break

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class encoder3(nn.Module):
    def __init__(self, vgg):
        super(encoder3, self).__init__()
        self.architecture = encoder_architecture
        self.network = self._create_conv_layers(self.architecture, vgg)

    def _create_conv_layers(self, architecture, vgg):
        layers = []

        for x in architecture:
            if type(x) == list:
                reflec = x[0]
                conv_layers = x[1]
                weight_index = x[2]
                activation = x[3]

                if reflec:
                    layers += [nn.ReflectionPad2d((1,1,1,1))]
                
                conv = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=conv_layers[2], stride=conv_layers[3], padding=conv_layers[4])
                conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(weight_index)].float())
                conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(weight_index)].float())
                layers += [conv]

                if activation:
                    layers += [nn.ReLU(inplace=True)]

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif x == "encoder 3 end":
                break

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class encoder4(nn.Module):
    def __init__(self, vgg):
        super(encoder4, self).__init__()
        self.architecture = encoder_architecture
        self.network = self._create_conv_layers(self.architecture, vgg)

    def _create_conv_layers(self, architecture, vgg):
        layers = []

        for x in architecture:
            if type(x) == list:
                reflec = x[0]
                conv_layers = x[1]
                weight_index = x[2]
                activation = x[3]

                if reflec:
                    layers += [nn.ReflectionPad2d((1,1,1,1))]
                
                conv = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=conv_layers[2], stride=conv_layers[3], padding=conv_layers[4])
                conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(weight_index)].float())
                conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(weight_index)].float())
                layers += [conv]

                if activation:
                    layers += [nn.ReLU(inplace=True)]

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif x == "encoder 4 end":
                break

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class encoder5(nn.Module):
    def __init__(self, vgg):
        super(encoder5, self).__init__()
        self.architecture = encoder_architecture
        self.network = self._create_conv_layers(self.architecture, vgg)

    def _create_conv_layers(self, architecture, vgg):
        layers = []

        for x in architecture:
            if type(x) == list:
                reflec = x[0]
                conv_layers = x[1]
                weight_index = x[2]
                activation = x[3]

                if reflec:
                    layers += [nn.ReflectionPad2d((1,1,1,1))]
                
                conv = nn.Conv2d(in_channels=conv_layers[0], out_channels=conv_layers[1], kernel_size=conv_layers[2], stride=conv_layers[3], padding=conv_layers[4])
                conv.weight = torch.nn.Parameter(vgg['{}.weight'.format(weight_index)].float())
                conv.bias = torch.nn.Parameter(vgg['{}.bias'.format(weight_index)].float())
                layers += [conv]

                if activation:
                    layers += [nn.ReLU(inplace=True)]

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif x == "encoder 5 end":
                break

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


"""class encoder1(nn.Module):
    def __init__(self, vgg):
        super(encoder1, self).__init__()
        self.conv1 = RCR2d(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = RCR2d(vgg, 2, 3, 64, 3, 1, 0, True, True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class encoder2(nn.Module):
    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.conv1 = RCR2d(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = RCR2d(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = RCR2d(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv4 = RCR2d(vgg, 9, 64, 128, 3, 1, 0, True, True)

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
        self.conv1 = RCR2d(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = RCR2d(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = RCR2d(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = RCR2d(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = RCR2d(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = RCR2d(vgg, 16, 128, 256, 3, 1, 0, True, True)

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
        self.conv1 = RCR2d(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = RCR2d(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = RCR2d(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = RCR2d(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = RCR2d(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = RCR2d(vgg, 16, 128, 256, 3, 1, 0, True, True)

        self.conv7 = RCR2d(vgg, 19, 256, 256, 3, 1, 0, True, True)

        self.conv8 = RCR2d(vgg, 22, 256, 256, 3, 1, 0, True, True)
        
        self.conv9 = RCR2d(vgg, 25, 256, 256, 3, 1, 0, True, True)

        # 28 x 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv10 = RCR2d(vgg, 29, 256, 512, 3, 1, 0, True, True)

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
        self.conv1 = RCR2d(vgg, 0, 3, 3, 1, 1, 0, False, False)

        self.conv2 = RCR2d(vgg, 2, 3, 64, 3, 1, 0, True, True)

        self.conv3 = RCR2d(vgg, 5, 64, 64, 3, 1, 0, True, True)

        # 112 x 112
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = RCR2d(vgg, 9, 64, 128, 3, 1, 0, True, True)

        self.conv5 = RCR2d(vgg, 12, 128, 128, 3, 1, 0, True, True)

        # 56 x 56
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv6 = RCR2d(vgg, 16, 128, 256, 3, 1, 0, True, True)

        self.conv7 = RCR2d(vgg, 19, 256, 256, 3, 1, 0, True, True)

        self.conv8 = RCR2d(vgg, 22, 256, 256, 3, 1, 0, True, True)
        
        self.conv9 = RCR2d(vgg, 25, 256, 256, 3, 1, 0, True, True)

        # 28 x 28
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv10 = RCR2d(vgg, 29, 256, 512, 3, 1, 0, True, True)

        self.conv11 = RCR2d(vgg, 32, 512, 512, 3, 1, 0, True, True)

        self.conv12 = RCR2d(vgg, 35, 512, 512, 3, 1, 0, True, True)

        self.conv13 = RCR2d(vgg, 38, 512, 512, 3, 1, 0, True, True)

        # 14 x 14
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv14 = RCR2d(vgg, 42, 512, 512, 3, 1, 0, True, True)

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

        return x"""


## Decoder
class decoder1(nn.Module):
    def __init__(self, dvgg):
        super(decoder1, self).__init__()
        self.conv3 = RCR2d(dvgg, 1, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv3(x)

        return x

class decoder2(nn.Module):
    def __init__(self, dvgg):
        super(decoder2, self).__init__()
        # 112 x 112
        self.conv5 = RCR2d(dvgg, 1, 128, 64, 3, 1, 0, True, True)
        
        # 224 x 224
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv6 = RCR2d(dvgg, 5, 64, 64, 3, 1, 0, True, True)

        self.conv7 = RCR2d(dvgg, 8, 64, 3, 3, 1, 0, True, False)

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
        self.conv7 = RCR2d(dvgg, 1, 256, 128, 3, 1, 0, True, True)

        # 112 x 112
        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv8 = RCR2d(dvgg, 5, 128, 128, 3, 1, 0, True, True)

        self.conv9 = RCR2d(dvgg, 8, 128, 64, 3, 1, 0, True, True)

        # 224 x 224
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv10 = RCR2d(dvgg, 12, 64, 64, 3, 1, 0, True, True)

        self.conv11 = RCR2d(dvgg, 15, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv7(x)

        x = self.unpool1(x)

        x = self.conv8(x)
        x = self.conv9(x)

        x = self.unpool2(x)

        x = self.conv10(x)
        x = self.conv11(x)

        return x

class decoder4(nn.Module):
    def __init__(self, dvgg):
        super(decoder4, self).__init__()
        # 28 x 28
        self.conv11 = RCR2d(dvgg, 1, 512, 256, 3, 1, 0, True, True)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv12 = RCR2d(dvgg, 5, 256, 256, 3, 1, 0, True, True)

        self.conv13 = RCR2d(dvgg, 8, 256, 256, 3, 1, 0, True, True)

        self.conv14 = RCR2d(dvgg, 11, 256, 256, 3, 1, 0, True, True)

        self.conv15 = RCR2d(dvgg, 14, 256, 128, 3, 1, 0, True, True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv16 = RCR2d(dvgg, 18, 128, 128, 3, 1, 0, True, True)

        self.conv17 = RCR2d(dvgg, 21, 128, 64, 3, 1, 0, True, True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv18 = RCR2d(dvgg, 25, 64, 64, 3, 1, 0, True, True)

        self.conv19 = RCR2d(dvgg, 28, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv11(x)

        x = self.unpool1(x)

        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.unpool2(x)

        x = self.conv16(x)
        x = self.conv17(x)

        x = self.unpool3(x)

        x = self.conv18(x)
        x = self.conv19(x)

        return x

class decoder5(nn.Module):
    def __init__(self, dvgg):
        super(decoder5, self).__init__()
        #
        self.conv15 = RCR2d(dvgg, 1, 512, 512, 3, 1, 0, True, True)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv16 = RCR2d(dvgg, 5, 512, 512, 3, 1, 0, True, True)
        
        self.conv17 = RCR2d(dvgg, 8, 512, 512, 3, 1, 0, True, True)
        
        self.conv18 = RCR2d(dvgg, 11, 512, 512, 3, 1, 0, True, True)
        
        self.conv19 = RCR2d(dvgg, 14, 512, 256, 3, 1, 0, True, True)

        #
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv20 = RCR2d(dvgg, 18, 256, 256, 3, 1, 0, True, True)
        
        self.conv21 = RCR2d(dvgg, 21, 256, 256, 3, 1, 0, True, True)
        
        self.conv22 = RCR2d(dvgg, 24, 256, 256, 3, 1, 0, True, True)
        
        self.conv23 = RCR2d(dvgg, 27, 256, 128, 3, 1, 0, True, True)

        #
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv24 = RCR2d(dvgg, 31, 128, 128, 3, 1, 0, True, True)
        
        self.conv25 = RCR2d(dvgg, 34, 128, 64, 3, 1, 0, True, True)

        #
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv26 = RCR2d(dvgg, 38, 64, 64, 3, 1, 0, True, True)
        
        self.conv27 = RCR2d(dvgg, 41, 64, 3, 3, 1, 0, True, False)

    def forward(self, x):
        x = self.conv15(x)
        
        x = self.unpool1(x)

        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)

        x = self.unpool2(x)

        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)

        x = self.unpool3(x)

        x = self.conv24(x)
        x = self.conv25(x)

        x = self.unpool4(x)

        x = self.conv26(x)
        x = self.conv27(x)

        return x