import torch
import torch.nn as nn

# Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, input):
        # batch, channel, height, width
        b, c, h, w = input.size()
        F = input.view(b, c, h*w)

        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)

        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        output = nn.MSELoss()(GramMatrix()(input), target)

        return output