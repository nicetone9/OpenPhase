import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
    """
    Extract coefficients at specified timesteps, then reshape for broadcasting.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels, cond):
        """
        Algorithm 1: Forward diffusion training.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        eps = self.model(x_t, t, labels, cond)
        loss = F.mse_loss(eps, noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, args):
        super().__init__()
        self.model = model
        self.T = T
        self.w = args.dif_w  # classifier-free guidance weight

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels, cond):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, labels, cond)
        null_labels = torch.zeros_like(labels).to(labels.device)
        non_eps = self.model(x_t, t, null_labels, cond)

        eps = (1. + self.w) * eps - self.w * non_eps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels, cond):
        """
        Algorithm 2: Reverse sampling.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t, t, labels, cond)

            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.sqrt(var) * noise

            assert torch.isnan(x_t).int().sum() == 0, "NaN detected in tensor."

        x_0 = torch.clip(x_t, -1, 1)
        return x_0

