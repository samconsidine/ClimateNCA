import torch
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ClimateNCA(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=128, fire_rate=1.0, kind='nca', activ_out='linear'):
        super().__init__()
        self.n_channels = n_channels
        self.device = device
        self.kind = kind

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=torch.float32,
        )
        filters = torch.stack(
                [identity_filter, sobel_filter_x, sobel_filter_y]
        )  
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(
                self.device
        )  

        if kind == 'cnn':
            if activ_out == 'tanh':
                out = nn.Tanh()
            elif activ_out == 'sigmoid':
                out = nn.Sigmoid()
            elif activ_out == 'linear':
                out = nn.Identity()

        # Update step
        self.update_module = nn.Sequential(
                nn.Conv2d(
                    3 * n_channels,
                    hidden_channels,
                    kernel_size=1,  # (1, 1)
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_channels,
                    n_channels,
                    kernel_size=1,
                    bias=False,
                ),
        )

        with torch.no_grad():
            self.update_module[2].weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        return nn.functional.conv2d(x, self.filters, padding=1, groups=self.n_channels)

    def update(self, x):
        return self.update_module(x)
    
    @staticmethod
    def get_living_mask(x):
        return (
            nn.functional.max_pool2d(
                x[:, :1, :, :], kernel_size=3, stride=1, padding=1
            )
            > 0.09
        )

    def forward(self, x, steps):
        for step in range(steps):
            pre_life_mask = self.get_living_mask(x)

            y = self.perceive(x)
            dx = self.update(y)
            dx = self.stochastic_update(dx, fire_rate=self.fire_rate)

            x = x + dx

            post_life_mask = self.get_living_mask(x)
            life_mask = (pre_life_mask & post_life_mask).to(torch.float32)

            x *= life_mask
            x = torch.clamp(x, min=0.0, max=1.0)

        return x