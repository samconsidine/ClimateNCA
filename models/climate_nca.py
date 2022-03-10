import torch
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CAModel(nn.Module):
    
    def __init__(self, num_channels, hidden_channels):
        super().__init__()

        self.num_channels = num_channels
        self.fire_rate = 1.0

        # sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        # sobel_x = sobel_filter_
        # sobel_y = sobel_filter_.t()
        # identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        # filters = torch.stack([identity_filter, sobel_x, sobel_y])
        # filters = filters.repeat((num_channels, 1, 1))
        # self.filters = filters[:, None, ...].to(device)

        self.update_module = nn.Sequential(
                nn.Conv2d(num_channels, hidden_channels, kernel_size=9, padding='same'),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, num_channels, kernel_size=9, bias=False, padding='same')
        )

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(device)

    def percieve(self, x):
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.num_channels)

    def update(self, x):
        return self.update_module(x)

    @staticmethod
    def stochastic_update(x, fire_rate):
        mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).to(device, torch.float32)
        return mask * x

    @staticmethod
    def get_living_mask(x):
        living_mask = nn.functional.max_pool2d( x[:, :1, :, :], kernel_size=3, stride=1, padding=1) > 0.32
        return nn.functional.max_pool2d( x[:, :1, :, :], kernel_size=3, stride=1, padding=1) > 0.32

    def forward(self, x):
        pre_life_mask = self.get_living_mask(x)

        # y = self.percieve(x)
        dx = self.update(x)
        dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

        x = x + dx

        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

        return x * life_mask

