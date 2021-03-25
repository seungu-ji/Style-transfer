import os
import argparse
from PIL import Image

from model import *
from utils import *

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser(description='Image Style Transfer Using Convolutional Neural Networks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--content_image', default='./datasets/content/Tuebingen_Neckarfront.jpg', type=str, dest='content_image')
    parser.add_argument('--style_image', default='./datasets/style/vangogh_starry_night.jpg', type=str, dest='style_image')
    parser.add_argument('--weight_path', default='./weights/vgg_conv.pth', type=str, dest='weight_path')
    parser.add_argument('--img_size', default=512, type=int, dest='img_size')
    # histogram loss
    parser.add_argument('--histogram_steps', default=150, type=int, dest='histogram_steps')

    args = parser.parse_args()

    content_image = args.content_image
    style_image = args.style_image
    weight_path = args.weight_path
    img_size = args.img_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prep and post processing for images
    prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    def postp(tensor): # to clip results in the range [0,1]
        t = postpa(tensor)
        t[t>1] = 1    
        t[t<0] = 0
        img = postpb(t)
        return img


    ## Loading
    # Load image: (style image, content image)
    imgs = [Image.open(style_image), Image.open(content_image)]
    """# display
    for img in imgs:
        imshow(img)
        show()"""
    imgs = [prep(img) for img in imgs]
    imgs = [Variable(img.unsqueeze(0).to(device)) for img in imgs]
    style_image, content_image = imgs

    opt_img = Variable(content_image.data.clone(), requires_grad=True)

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
    if args.histogram_steps is not None:
      fn_loss += [HistogramLoss(num_steps=150, cuda=True)]
    fn_loss = [loss.to(device) for loss in fn_loss]

    if args.histogram_steps is not None:
        loss_layer = loss_layer + HistogramLoss(num_steps=args.histogram_steps, cuda=True)

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
    optim = torch.optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optim.zero_grad()
            output = net(opt_img, loss_layer)
            layer_losses = [weights[a] * fn_loss[a](A, targets[a]) for a, A in enumerate(output)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            # print loss
            if n_iter[0] % show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f' % (n_iter[0]+1, loss.item()))
            
            return loss

        optim.step(closure)

    output = postp(opt_img.data[0].cpu().squeeze())
    output.save("output.jpg", "JPEG", quality=80, optimize=True, progressive=True)
    """
    imshow(output)
    gcf().set_size_inches(10,10)
    """

if __name__ == '__main__':
    main()