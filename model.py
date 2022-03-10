import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CAModel(nn.Module):
    def __init__(self, num_channels, hidden_channels, fire_rate=0.5, device='cpu', hidden_size=128):
        super(CAModel, self).__init__()

        channel_n = num_channels
        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Conv2d(channel_n*3, hidden_size, kernel_size=9, padding='same')
        self.fc1 = nn.Conv2d(hidden_size, 1, bias=False, kernel_size=9, padding='same')
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, :, :, :], kernel_size=3, stride=1, padding=1) > 0.3260026

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        #x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx#.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # self.layer1 = nn.Linear(in_features=12 * 128 * 128, out_features=256)
#         # self.layer2 = nn.Linear(in_features=256, out_features=256)
#         # self.layer3 = nn.Linear(in_features=256, out_features=24 * 64 * 64)
#         self.layer1 = nn.Conv2d(3, 6, 5)
#         self.layer2 = nn.Conv2d(3, 6, 5)

#     def forward(self, features):
#         x = features.view(-1, 12 * 128 * 128) / 1024.0
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.relu(self.layer3(x))

#         return x.view(-1, 24, 64, 64) * 1024.0
