from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        A residual block for 1D convolutional data.
        Args:
            in_channels (int):  Number of input channels.
            out_channels (int): Number of output channels.
            stride (int):       Controls the stride.
        """
        super(Block, self).__init__()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, batch):
        """
        Args:
            batch (dict): Dictionary with at least the 'x' key: input tensor of shape [B, C, L]
        Returns:
            Tensor of shape [B, out_channels, L] or similar
        """
        x = batch['x']  # Extract input features
        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out
