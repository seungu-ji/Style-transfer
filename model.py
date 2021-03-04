import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self, pool_type='max'):
        super(VGG19, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool_type == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, output_keys):
        output = {}

        output['c11'] = F.relu(self.conv1_1(x))
        output['c12'] = F.relu(self.conv2_1(output['c11']))
        output['p1'] = self.pool1(output['c12'])

        output['c21'] = F.relu(self.conv2_1(output['p1']))
        output['c22'] = F.relu(self.conv2_1(output['c21']))
        output['p2'] = self.pool1(output['c22'])

        output['c31'] = F.relu(self.conv2_1(output['p2']))
        output['c32'] = F.relu(self.conv2_1(output['c31']))
        output['c33'] = F.relu(self.conv2_1(output['c32']))
        output['c34'] = F.relu(self.conv2_1(output['c33']))
        output['p3'] = self.pool1(output['c34'])

        output['c41'] = F.relu(self.conv2_1(output['p3']))
        output['c42'] = F.relu(self.conv2_1(output['c41']))
        output['c43'] = F.relu(self.conv2_1(output['c42']))
        output['c44'] = F.relu(self.conv2_1(output['c43']))
        output['p4'] = self.pool1(output['c44'])
        
        output['c51'] = F.relu(self.conv2_1(output['p4']))
        output['c52'] = F.relu(self.conv2_1(output['c51']))
        output['c53'] = F.relu(self.conv2_1(output['c52']))
        output['c54'] = F.relu(self.conv2_1(output['c53']))
        output['p5'] = self.pool1(output['c54'])
        
        return [output[key] for key in output_keys]