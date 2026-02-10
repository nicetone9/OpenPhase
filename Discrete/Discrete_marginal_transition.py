import torch

class DiscreteMarginalTransition:
    def __init__(self, x_classes=21, x_marginal_path='data/train_marginal_x.pt'):
        self.X_classes = x_classes
        self.x_marginal = torch.load(x_marginal_path)
        if self.x_marginal.dim() == 3 and self.x_marginal.shape[0] == 1:
            self.x_marginal = self.x_marginal.squeeze(0)

        assert self.x_marginal.dim() == 2 and self.x_marginal.shape[1] == self.X_classes, \
            f"Expected x_marginal shape [L, {self.X_classes}], got {self.x_marginal.shape}"
        avg_marginal = self.x_marginal.mean(dim=0)  # shape: (V,)
        self.u_x = avg_marginal.unsqueeze(0).repeat(self.X_classes, 1)  # shape: (V, V)

    def get_Qt(self, beta_t, device):
        """
        Compute one-step transition matrix Q_t from x_{t-1} to x_t.
        Args:
            beta_t: (B,) tensor of betas for each sample in batch
        Returns:
            Q_t: (B, V, V) transition matrices
        """
        B = beta_t.shape[0]
        beta_t = beta_t.to(device).view(B, 1, 1)     # shape: (B, 1, 1)
        u_x = self.u_x.to(device).unsqueeze(0)       # shape: (1, V, V)
        I = torch.eye(self.X_classes, device=device).unsqueeze(0)  # shape: (1, V, V)

        Q_t = beta_t * u_x + (1 - beta_t) * I        # shape: (B, V, V)
        return Q_t

    def get_Qt_bar(self, alpha_bar_t, device):
        """
        Compute t-step marginal transition matrix Q̄_t from x₀ to x_t.
        Args:
            alpha_bar_t: (B,) tensor of cumulative alphas
        Returns:
            Q̄_t: (B, V, V) transition matrices
        """
        B = alpha_bar_t.shape[0]
        alpha_bar_t = alpha_bar_t.to(device).view(B, 1, 1)  # shape: (B, 1, 1)
        u_x = self.u_x.to(device).unsqueeze(0)              # shape: (1, V, V)
        I = torch.eye(self.X_classes, device=device).unsqueeze(0)  # shape: (1, V, V)

        Q_t_bar = alpha_bar_t * I + (1 - alpha_bar_t) * u_x  # shape: (B, V, V)
        return Q_t_bar