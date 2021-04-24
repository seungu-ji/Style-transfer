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


