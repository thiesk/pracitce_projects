import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(F.relu(x))

        return x

class Convolution_net(nn.Module):
    '''
    Stupid simple convolutional network.
    Some convolution layers and fc classification head.
    '''
    def __init__(self):
        super(Convolution_net, self).__init__()

        self.cb1 = conv_block(1, 32, 3, 1, 'same')
        self.cb2 = conv_block(32, 16, 3, 1, 'same')

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)



    def forward(self, x):
        # first conv block
        x = self.cb1(x)
        x = F.max_pool2d(x,2,2)
        x = nn.Dropout(0.2)(x)

        # second conv block
        x = self.cb2(x)
        x = F.max_pool2d(x,2,2)
        x = nn.Dropout(0.2)(x)

        # fc layer for classification
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
