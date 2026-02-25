
import torch
from torch import nn
import torch.nn.functional as F

# The following section is adapted from the PRO-LDM repo:

# https://github.com/AzusaXuan/PRO-LDM

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Residual Block for 1D Convolution.

        Args:
            in_channels (int):  Number of input channels.
            out_channels (int): Number of output channels.
            stride (int):       Stride for the first convolution. Default: 1.
        """
        super(Block, self).__init__()

        self.use_skip = stride != 1 or in_channels != out_channels

        # optional
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if self.use_skip else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x) if self.use_skip else x
        out = self.block(x)
        out += identity
        return F.relu(out)