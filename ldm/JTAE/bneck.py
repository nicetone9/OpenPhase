
from torch import nn
# The following section is adapted from the PRO-LDM repo:
# https://github.com/AzusaXuan/PRO-LDM

class BaseBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()
        self.linear = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))