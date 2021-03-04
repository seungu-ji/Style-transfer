import os
import argparse
from PIL import Image

from model import *
from utils import *

import torch
import torch.nn as nn
from torch import optim

def main():
    parser = argparse.ArgumentParser(description='Image Style Transfer Using Convolutional Neural Networks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--content_image", default="./dataset/content.jpg", type=str, dest="content_image")
    parser.add_argument("--style_image", default="./dataset/style.jpg", type=str, dest="style_image")
    parser.add_argument("--weight_path", default="./weight/     ", type=str, dest="weight_path")
    

    args = parser.parse_args()

    content_image = args.content_image
    style_image = args.style_image
    weight_path = args.weight_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained VGGNet
    net = VGG19().to(device)
    net.load_state_dict(torch.load(weight_path))

    for param in net.parameters():
        param.requires_grad = False

    ## Style transfer setting
    style_layers = ['c11', 'c21', 'c31', 'c41', 'c51']
    content_layer = ['c42']

    # loss setting
    loss_layer = style_layers + content_layer
    fn_loss = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layer)
    fn_loss = [loss.to(device) for loss in fn_loss]

    # weight setting
    style_weight = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weight = [1e0]
    weights = style_weight + content_weight

    # compute optimization target
    style_target = [GramMatrix()(A).detach() for A in net(style_image, style_layers)]
    content_target = [A.detach() for A in net(content_image, content_layer)]
    targets = style_target + content_target

    ## Style transfer
    max_iter = 500
    show_iter = 50
    optim = optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optim.zero_grad()
            output = net(opt_img, loss_layer)
            layer_losses = [weights[a] * fn_loss[a](A, targets[a]) for a, A in enumerate(output)]
            loss = sum(layer_losses)
            loss.backward()
            pass


if __name__ == '__main__':
    main()