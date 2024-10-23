import torch
import torch.nn as nn

class convolution_net(nn.Module):
    def __init__(self):
        self.resolution = 28 * 28
        self.cl1 = nn.Conv2d(1, 21, 3)
        self.cl2 = nn.Conv2d(32, 16, 3)
        self.cl3 = nn.Conv2d(16, 8, 3)

        self.fc1 = nn.Linear(8 * self.resolution, 100)
        self.fc2 = torch.Linear(100, 10)
