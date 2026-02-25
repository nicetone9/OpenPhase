import torch
import torch.nn as nn
import wandb

# The following section is adapted from the PRO-LDM repo:
# https://github.com/AzusaXuan/PRO-LDM

def loss_function(args, predictions, targets, z_rep, diff_loss):
    """
    Computes the total loss consisting of:
    - Cross-entropy loss between predictions and ground truth
    - L2 regularization loss on latent representation
    - Diffusion loss
    Logs all losses to WandB if enabled.

    Args:
        args: argparse.Namespace or object with necessary hyperparams
        predictions: model output logits (B, input_dim, seq_len)
        targets: true token indices (B, seq_len)
        z_rep: latent vector (B, latent_dim)
        diff_loss: scalar diffusion loss

    Returns:
        total_loss: summed loss
    """
    x_hat = predictions
    x_true = targets.to(device=x_hat.device)
    seq_len = x_true.shape[1]

    ce_loss_weights = torch.ones(args.input_dim).to(x_hat.device)
    ce_loss_weights[21] = 0.8 
    ae_loss = nn.CrossEntropyLoss(weight=ce_loss_weights)(x_hat, x_true)
    ae_loss *= args.gamma_val
    zrep_l2_loss = 0.5 * torch.norm(z_rep, p=2, dim=1).pow(2).mean()
    zrep_l2_loss *= args.eta_val

    # Diffusion loss
    diff_loss *= args.sigma_val

    # Total
    total_loss = ae_loss + zrep_l2_loss + diff_loss

    # Logging metrics
    seq_difference = (x_true - x_hat.argmax(dim=1)).count_nonzero() / (args.batch_size * seq_len)
    if args.use_wandb:
        wandb.log({
            "seq difference": seq_difference.item(),
            "ae_loss": ae_loss.item(),
            "z_loss": zrep_l2_loss.item(),
            "diff_loss": diff_loss.item(),
            "total_loss": total_loss.item()
        })

    return total_loss