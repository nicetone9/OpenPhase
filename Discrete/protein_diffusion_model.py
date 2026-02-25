import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
from tqdm import tqdm
from Discrete_marginal_transition import DiscreteMarginalTransition
from utils_ import PredefinedNoiseScheduleDiscrete
from utils_ import get_entropy,fuse_logits_by_log_probs
from tensor_to_list import tensor_to_sequence_list
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
import esm
import traceback, inspect
# -------------------------
# Utility / small helpers
# -------------------------
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


# -------------------------
# Embeddings
# -------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        """
        d_model: embedding dim for sinusoidal positions (must be even)
        T: number of discrete timesteps for embedding table (used when passing integer timesteps)
        dim: output dimension for time embedding projections
        """
        assert d_model % 2 == 0
        super().__init__()
        # create sinusoidal table of shape (T, d_model)
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)  # (T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),  # accepts long indices 0..T-1
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t_idx):
        # expects long indices (B,) -> returns (B, dim)
        return self.timembedding(t_idx)


class ConditionalEmbedding(nn.Module):
    def __init__(self, cond_dim, tdim):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Linear(cond_dim, tdim),
            Swish(),
            nn.Linear(tdim, tdim),
        )

    def forward(self, c):
        return self.condEmbedding(c)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, tdim):
        super().__init__()
        self.labelEmbedding = nn.Sequential(
            nn.Embedding(num_classes, tdim),
            Swish(),
            nn.Linear(tdim, tdim)
        )

    def forward(self, y):
        # ensure indices are long
        if y is None:
            return None
        if y.dtype != torch.long:
            y = y.long()
        return self.labelEmbedding(y)


# -------------------------
# Simple 1D blocks
# -------------------------
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv1d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, *args):
        return self.c1(x) + self.c2(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose1d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, *args):
        x = self.t(x)
        return self.c(x)


# -------------------------
# Attention block (channel-wise)
# -------------------------
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

        # project & permute for attention
        q = self.proj_q(h).permute(0, 2, 1)  # [B, L, C]
        k = self.proj_k(h).permute(0, 2, 1)  # [B, L, C]
        v = self.proj_v(h).permute(0, 2, 1)  # [B, L, C]

        # attention weights: [B, L, L]
        w = torch.bmm(q, k.transpose(1, 2)) * (C ** -0.5)
        w = F.softmax(w, dim=-1)

        # output: [B, L, C]
        h = torch.bmm(w, v).permute(0, 2, 1)  # back to [B, C, L]

        return x + self.proj(h)


# -------------------------
# Residual block (with embedding handling)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1_norm = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.block1_conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        # projection MLPs for embeddings -> produce out_ch sized vectors
        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.cond_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.label_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))

        self.block2_norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.block2_act = Swish()
        self.block2_dropout = nn.Dropout(dropout)
        self.block2_conv = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def _squeeze_embedding(self, emb):
        """Accepts emb in [B, tdim] or [B, 1, tdim] and returns [B, tdim] or None."""
        if emb is None:
            return None
        if emb.dim() == 2:
            return emb
        if emb.dim() == 3 and emb.size(1) == 1:
            return emb.squeeze(1)
        # unexpected shape -> try to flatten trailing dims (defensive)
        if emb.dim() > 1:
            return emb.view(emb.size(0), -1)
        return emb

    def forward(self, x, temb, cemb, yemb):
        """
        x: [B, C, L]
        temb/cemb/yemb: either [B, tdim] or [B, 1, tdim] or None
        """
        # Normalize embeddings to [B, tdim] or None
        temb = self._squeeze_embedding(temb)
        cemb = self._squeeze_embedding(cemb)
        yemb = self._squeeze_embedding(yemb)

        # First conv block
        h = self.block1_norm(x)
        h = Swish()(h)
        h = self.block1_conv(h)  # -> [B, out_ch, L]

        # Project embeddings into channel space and unsqueeze to broadcast across length
        temb_p = self.temb_proj(temb).unsqueeze(-1) if temb is not None else 0
        cemb_p = self.cond_proj(cemb).unsqueeze(-1) if cemb is not None else 0
        yemb_p = self.label_proj(yemb).unsqueeze(-1) if yemb is not None else 0

        # Add and continue
        h = h + temb_p + cemb_p + yemb_p

        h = self.block2_norm(h)
        h = self.block2_act(h)
        h = self.block2_dropout(h)
        h = self.block2_conv(h)

        h = h + self.shortcut(x)
        return self.attn(h)


# -------------------------
# UNet (full)
# -------------------------
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout,
                 cond_dim=14, num_classes=21, debug=False):
        """
        T: number of discrete timesteps used by TimeEmbedding
        ch: base hidden channels (embedding dim)
        ch_mult: list of multiplers for down/up channels, e.g., [1,2,4]
        num_res_blocks: how many residual blocks per level
        """
        super().__init__()
        self.ch = ch
        self.debug = debug

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(cond_dim, tdim)
        self.label_embedding = LabelEmbedding(num_classes, tdim)

        # Token embedding (maps tokens -> ch dimensional vectors)
        self.token_embed = nn.Embedding(num_classes, ch)

        # head conv expects channels=ch
        self.head = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

        # build down path
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

        # middle
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        # up path
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False)
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # tail
        self.tail = nn.Sequential(
            nn.GroupNorm(min(8, now_ch), now_ch),
            Swish(),
            nn.Conv1d(now_ch, num_classes, kernel_size=3, padding=1)
        )

    def _time_to_index(self, t, device):
        """
        Accept t as either
          - float tensor in [0,1] shape (B,) or (B,1)
          - long/int tensor indices in [0, T-1]
        Return long tensor indices shape (B,) for TimeEmbedding
        """
        # If float, map to discrete index range
        if t is None:
            return torch.zeros(1, dtype=torch.long, device=device)
        if t.dtype.is_floating_point:
            t_flat = t.view(-1)
            T = self.time_embedding.timembedding[0].num_embeddings
            idx = (t_flat.clamp(0.0, 1.0) * (T - 1)).long().to(device)
            return idx
        else:
            return t.view(-1).long().to(device)

    def forward(self, x, t, c=None, y=None):
        """
        x: [B, L] token indices (LongTensor)
        t: either float normalized [0,1] Tensor (B,) or LongTensor indices (B,)
        c: condition vector [B, cond_dim] (float)
        y: class labels [B] (int/long)
        """
        if self.debug:
            print(f"[Input tokens] {x.shape}")

        # Embed tokens -> [B, L, ch], then permute -> [B, ch, L]
        x = self.token_embed(x)  # [B, L, ch]
        if self.debug:
            print(f"[Token embed] {x.shape}")
        x = x.permute(0, 2, 1).contiguous()  # [B, ch, L]
        if self.debug:
            print(f"[Token permute] {x.shape}")

        # prepare embeddings
        device = x.device
        t_idx = self._time_to_index(t, device)            # [B]
        temb = self.time_embedding(t_idx)                 # [B, tdim]
        cemb = self.cond_embedding(c) if c is not None else None
        yemb = self.label_embedding(y) if y is not None else None
        if self.debug:
            print(f"[temb] {temb.shape}, [cemb] {getattr(cemb, 'shape', None)}, [yemb] {getattr(yemb, 'shape', None)}")

        # Down path
        h = self.head(x)
        if self.debug:
            print(f"[Head conv out] {h.shape}")
        hs = [h]
        for i, layer in enumerate(self.downblocks):
            if isinstance(layer, ResBlock):
                if self.debug: print(f"[Down {i} in] {h.shape}")
                h = layer(h, temb, cemb, yemb)
                if self.debug: print(f"[Down {i} out] {h.shape}")
            else:
                h = layer(h)
                if self.debug: print(f"[DownSample {i} out] {h.shape}")
            hs.append(h)

        # Middle
        for i, layer in enumerate(self.middleblocks):
            if self.debug: print(f"[Middle {i} in] {h.shape}")
            h = layer(h, temb, cemb, yemb)
            if self.debug: print(f"[Middle {i} out] {h.shape}")

        # Up path
        for i, layer in enumerate(self.upblocks):
            if isinstance(layer, ResBlock):
                skip = hs.pop()
                if self.debug: print(f"[Up {i} concat] h={h.shape}, skip={skip.shape}")
                h = torch.cat([h, skip], dim=1)
                if self.debug: print(f"[Up {i} after cat] {h.shape}")
                h = layer(h, temb, cemb, yemb)
                if self.debug: print(f"[Up {i} out] {h.shape}")
            else:
                h = layer(h)
                if self.debug: print(f"[UpSample {i} out] {h.shape}")
        assert len(hs) == 0

        if self.debug: print(f"[Tail in] {h.shape}")
        out = self.tail(h)  # [B, num_classes, L]
        if self.debug: print(f"[Tail out] {out.shape}")
        return out.permute(0, 2, 1).contiguous()  # [B, L, num_classes]

class ProteinDiffusionModel(nn.Module):
    def __init__(
        self, model, prior_model, esm_batch_converter,
        timesteps, objective='pred_x0', noise_type='marginal', loss_type='CE',
        marginal_dist_path='data/train_marginal_x.pt'
    ):
        super().__init__()
        self.model = model
        self.prior_model = prior_model
        self.loss_type = loss_type
        self.esm_batch_converter = esm_batch_converter

        if noise_type == 'marginal':
            self.transition_model = DiscreteMarginalTransition(
                x_classes=21, x_marginal_path=marginal_dist_path
            )
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule='cosine', timesteps=timesteps
        )

        self.timesteps = timesteps
        self.objective = objective

    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy
        else:
            raise ValueError(f"Unknown loss_type {self.loss_type}")

    def _ensure_indices(self, x):
        """Convert [B, L, 21] one-hot or [B, L] int -> [B, L] int"""
        if x.dim() == 3 and x.shape[-1] == 21:
            return x.argmax(dim=-1).long()
        return x.long()

    def _to_onehot(self, x_indices):
        """Convert [B, L] int -> [B, L, 21] float one-hot"""
        return F.one_hot(x_indices, num_classes=21).float()

    def apply_noise(self, x, t_int):
        """
        Apply discrete marginal noise.
        Args:
            x: [B, L] int or [B, L, 21] one-hot
            t_int: [B, 1] int timesteps
        Returns:
            noise_onehot: [B, L, 21] float
            noise_indices: [B, L] long
            alpha_t_bar: [B, 1] float
        """
        x_idx = self._ensure_indices(x)
        B, L = x_idx.shape

        t_float = t_int / self.timesteps
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=x.device)

        x_onehot = self._to_onehot(x_idx)
        x_flat = x_onehot.view(B * L, 1, 21)
        Qtb_flat = Qtb.repeat_interleave(L, dim=0)
        Qtb_mat = Qtb_flat.squeeze(1)
        prob_X = torch.bmm(x_flat, Qtb_mat.transpose(1, 2)).squeeze(1)  # [B*L, 21]
        X_t = prob_X.multinomial(1).squeeze(-1)                          # [B*L]
        noise_indices = X_t.view(B, L)
        noise_onehot = self._to_onehot(noise_indices)
        return noise_onehot, noise_indices, alpha_t_bar

    def diffusion_loss(self, x, t_int):
        x_idx = self._ensure_indices(x)
        B, L = x_idx.shape
        device = x.device

        s_int = t_int - 1
        t_float = t_int / self.timesteps
        s_float = torch.clamp(s_int / self.timesteps, 0, 1)

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=device)
        Qt = self.transition_model.get_Qt(beta_t, device=device)

        x_onehot = self._to_onehot(x_idx)
        x_flat = x_onehot.view(B * L, 1, 21)
        Qtb_flat = Qtb.repeat_interleave(L, dim=0)
        Qtb_mat = Qtb_flat.squeeze(1)
        prob_X = torch.bmm(x_flat, Qtb_mat.transpose(1, 2)).squeeze(1)
        X_t = prob_X.multinomial(1).squeeze(-1)
        noise_indices = X_t.view(B, L)
        noise_onehot = self._to_onehot(noise_indices)
        return noise_onehot, noise_indices

    def compute_val_loss(self, x):
        B = x.shape[0]
        t_int = torch.randint(1, self.timesteps + 1, size=(B, 1), device=x.device)
        return self.diffusion_loss(x, t_int)

    def forward(self, x, c=None, y=None):
        B = x.shape[0]
        t_int = torch.randint(0, self.timesteps + 1, size=(B, 1), device=x.device)
        noise_onehot, noise_indices, _ = self.apply_noise(x, t_int)

        if self.objective != 'pred_x0':
            raise ValueError(f'Unknown objective: {self.objective}')

        # Pass indices to the base model
        base_logits = self.model(noise_indices, t_int, c=c, y=y)  # [B, L, 21]

        log_probs = F.log_softmax(base_logits, dim=-1)
        base_pred_x = self._to_onehot(log_probs.argmax(dim=-1))

        entropy = get_entropy(log_probs)
        mask_entropy = entropy > torch.quantile(entropy, 0.9)

        masked_input = base_pred_x.clone()
        masked_input[mask_entropy] = 0

        seqs = tensor_to_sequence_list(masked_input, mask_entropy)
        batch_data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = self.esm_batch_converter(batch_data)
        tokens = tokens.to(masked_input.device)

        with torch.no_grad():
            esm_out = self.prior_model(tokens)
            esm_logits = esm_out["logits"][:, 1:-1, :21]
        prior_logits = esm_logits

        target_class = self._ensure_indices(x)
        loss_fn = self.loss_fn()
        base_loss = loss_fn(base_logits.view(-1, 21), target_class.view(-1))
        mask_loss = loss_fn(prior_logits[mask_entropy], target_class[mask_entropy])

        return base_loss, mask_loss

    def compute_batched_over0_posterior_distribution(self, X_t_onehot, Q_t, Qsb, Qtb, batch):
        Qt_T = Q_t.transpose(-1, -2)
        left_term = torch.matmul(X_t_onehot, Qt_T[batch])
        left_term_exp = left_term.unsqueeze(2)
        Qsb_exp = Qsb[batch].unsqueeze(1)
        numerator = left_term_exp * Qsb_exp

        X_t_T = X_t_onehot.unsqueeze(-1)
        Qtb_exp = Qtb[batch].unsqueeze(1)
        denominator = torch.matmul(Qtb_exp, X_t_T).squeeze(-1)
        denominator[denominator == 0] = 1e-6

        numerator = numerator.sum(dim=-1)
        posterior = numerator / denominator
        return posterior
    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def mc_ddim_sample(
        self, x=None, cond=None, y=None, seq_len=None, diverse=True, stop=0, step=50
    ):
        device = next(self.parameters()).device
        B = x.shape[0] if x is not None else 1

        if seq_len is None:
            if x is None:
                raise ValueError("Either x or seq_len must be provided")
            seq_len = x.shape[1]

        limit_dist = self.transition_model.x_marginal.view(-1)
        zt_idx = torch.multinomial(limit_dist, seq_len, replacement=True)
        zt_idx = zt_idx.unsqueeze(0).repeat(B, 1).to(device)
        zt_idx = torch.clamp(zt_idx, 0, 20)

        last_logits = None

        for s_int in reversed(range(stop, self.timesteps, step)):
            s_norm = torch.tensor([[s_int]], device=device) / self.timesteps
            t_norm = torch.tensor(
                [[min(s_int + step, self.timesteps)]], device=device
            ) / self.timesteps

            last_logits, zt_idx = self.sample_p_zs_given_zt(
                t=t_norm,
                s=s_norm,
                zt=zt_idx,
                x=x,
                diverse=diverse,
                sample_type="ddim",
                last_step=(s_int == 0),
                c=cond,
                y=y,
            )

        final_idx = torch.clamp(self._ensure_indices(zt_idx), 0, 20)
        #print(final_idx.shape, final_idx.dtype)
        final_onehot = self._to_onehot(final_idx)
        seqs = tensor_to_sequence_list(final_idx)

        return last_logits, final_onehot, seqs

    # ------------------------------------------------------------------
    # Core DDIM step
    # ------------------------------------------------------------------
    def sample_p_zs_given_zt(
        self, t, s, zt, x=None, diverse=True, sample_type="ddim",
        last_step=False, c=None, y=None
    ):
        device = next(self.parameters()).device

        zt_idx = torch.clamp(self._ensure_indices(zt), 0, 20).to(device)
        B, L = zt_idx.shape

        zt_onehot = self._to_onehot(zt_idx)
        V = zt_onehot.size(-1)

        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=device)

        if sample_type == "ddpm":
            Qt = self.transition_model.get_Qt(beta_t, device=device)
        else:
            Qt = Qsb / (Qtb + 1e-12)
            Qt = Qt / (Qt.sum(dim=-1, keepdim=True) + 1e-12)

        logits_base = self.model(zt_idx, t, c=c, y=y)
        log_probs = F.log_softmax(logits_base, dim=-1)

        entropy = get_entropy(log_probs)
        mask_entropy = entropy > torch.quantile(entropy, 0.9)

        safe_idx = log_probs.argmax(dim=-1)
        safe_idx[mask_entropy] = 0

        safe_onehot = self._to_onehot(safe_idx)
        seq_strings = tensor_to_sequence_list(safe_onehot)

        batch_data = [(f"seq{i}", s) for i, s in enumerate(seq_strings)]
        _, _, tokens = self.esm_batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            esm_out = self.prior_model(tokens)
            esm_logits = esm_out["logits"][:, 1:-1, :V]
            prior_log_probs = F.log_softmax(esm_logits, dim=-1)

        logits = fuse_logits_by_log_probs(
            [log_probs, prior_log_probs],
            [logits_base, esm_logits],
        )
        pred_X = F.softmax(logits, dim=-1)

        if last_step:
            return logits, pred_X.argmax(dim=-1)

        batch_idx = torch.arange(B, device=device)
        p_sXt = self.compute_batched_over0_posterior_distribution(
            zt_onehot, Qt, Qsb, Qtb, batch_idx
        )
        p_sXt = p_sXt / (p_sXt.sum(dim=-1, keepdim=True) + 1e-12)

        prob_X = pred_X * p_sXt
        prob_X = prob_X / (prob_X.sum(dim=-1, keepdim=True) + 1e-12)

        if diverse:
            sample_s = prob_X.reshape(-1, V).multinomial(1).view(B, L)
        else:
            sample_s = prob_X.argmax(dim=-1)

        return logits, sample_s



