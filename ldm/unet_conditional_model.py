import math
import torch
from torch import nn
from torch.nn import functional as F


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.timembedding(t)


class ConditionEmbedding(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, dim),
            Swish(),
            nn.Linear(dim, dim)
        )

    def forward(self, c):
        return self.embedding(c)


class LabelEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, dim),
            Swish(),
            nn.Linear(dim, dim)
        )

    def forward(self, y):
        return self.embedding(y.view(-1, 1).float())


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv1d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb, yemb):
        return self.c1(x) + self.c2(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose1d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb, yemb):
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1)
        self.proj = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, L = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 1)
        k = self.proj_k(h)
        v = self.proj_v(h).permute(0, 2, 1)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=-1)
        h = torch.bmm(w, v).permute(0, 2, 1)
        return x + self.proj(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.cemb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.yemb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb, cemb, yemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None]
        h += self.cemb_proj(cemb)[:, :, None]
        h += self.yemb_proj(yemb)[:, :, None]
        h = self.block2(h) + self.shortcut(x)
        return self.attn(h)


class UNet(nn.Module):
    def __init__(self, T, condition_dim, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionEmbedding(condition_dim, tdim)
        self.label_embedding = LabelEmbedding(tdim)
        self.head = nn.Conv1d(1, ch, 3, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv1d(now_ch, 1, 3, padding=1),
        )

    def forward(self, x, t, c, y):
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(c)
        yemb = self.label_embedding(y)

        h = self.head(x.unsqueeze(1))
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb, yemb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb, cemb, yemb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb, yemb)

        return self.tail(h).squeeze(1)
