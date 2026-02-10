from torch import nn

class BaseBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, batch):
        """
        batch: dict with keys 'x', 'c', 'y'
        - x: input features [B, input_dim]
        - c: condition
        - y: label 
        """
        x = batch['x']  # Extract input
        z_rep = self.fc1(x)
        return z_rep



