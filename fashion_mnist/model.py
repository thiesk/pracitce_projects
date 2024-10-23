import torch.nn as nn
import torch.nn.functional as F

class Convolution_net(nn.Module):
    '''
    Stupid simple convolutional network.
    Some convolution layers and fc classification head.
    '''
    def __init__(self):
        super(Convolution_net, self).__init__()

        self.cl1 = nn.Conv2d(1, 32, 3)
        self.cl2 = nn.Conv2d(32, 16, 3)

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 10)



    def forward(self, x):
        # first conv block
        x = self.cl1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # second conv block
        x = self.cl2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        # fc layer for classification
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
