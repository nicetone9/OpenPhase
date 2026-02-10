import torch
from torch import nn
import argparse
from model.JTAE.convolutional import Block
from model.ConDiff.DiffusionFreeGuidence.ModelCondition import UNet
from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer

class jtae(nn.Module):
    def __init__(self, hparams):
        super(jtae, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams
        self.input_dim = hparams.input_dim       # input = D_x + D_c
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.embedding_dim = hparams.embedding_dim
        self.seq_len = hparams.seq_len
        self.device = hparams.device

        # Core embedding layer (MLP over concatenated ESM + condition)
        self.embed = nn.Linear(self.input_dim, self.embedding_dim)

        # Optional bottleneck
        self.bottleneck = nn.Linear(self.embedding_dim, self.latent_dim)

        # Decoder
        self.dec_conv_module = nn.ModuleList([
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            Block(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1),
        ])

        # Diffusion model
        self.diff_model = UNet(
            T=hparams.dif_T,
            num_labels=hparams.num_labels,
            ch=hparams.dif_channel,
            ch_mult=hparams.dif_channel_mult,
            num_res_blocks=hparams.dif_res_blocks,
            dropout=hparams.dif_dropout
        )

        self.Gaussian_Diffusion_Trainer = GaussianDiffusionTrainer(
            model=self.diff_model,
            beta_1=hparams.dif_beta_1,
            beta_T=hparams.dif_beta_T,
            T=hparams.dif_T
        )

    def encode(self, fused):
        x = self.embed(fused)               # [B, embedding_dim]
        z_rep = self.bottleneck(x)         # [B, latent_dim]
        return z_rep

    def decode(self, z_rep):
        h = z_rep
        for i, layer in enumerate(self.dec_conv_module):
            if i == 1:
                h = h.reshape(-1, self.hidden_dim // 2, self.seq_len)
            h = layer(h)
        return h

    def diff_train(self, z_rep, labels):
        b = z_rep.shape[0]
        if torch.rand(1).item() < 0.1:
            labels = torch.zeros_like(labels)
        labels = labels.long()
        loss = self.Gaussian_Diffusion_Trainer(z_rep.unsqueeze(1), labels)
        return loss.sum() / (b ** 2)

    def forward_dict(self, batch_dict):
        x = batch_dict['x'].to(self.device)  # [B, D_x]  (ESM)
        c = batch_dict['c'].to(self.device)  # [B, D_c]  (Condition)
        y = batch_dict['y'].to(self.device)  # [B] or [B, 1]

        fused = torch.cat([x, c], dim=-1)    # [B, D_x + D_c]
        z_rep = self.encode(fused)           # [B, latent_dim]
        diff_loss = self.diff_train(z_rep, y)

        x_hat = self.decode(z_rep)           # [B, input_dim, seq_len] or similar

        return x_hat, z_rep, diff_loss
