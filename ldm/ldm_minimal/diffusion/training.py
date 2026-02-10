import os
import numpy as np
import torch
#from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer

def train(params, x, c, labels, net_model):
    device = params.device

    # Load model weights if provided
    if params.dif_training_load_weight is not None:
        net_model.load_state_dict(
            torch.load(os.path.join(params.dif_sav_dir, params.dif_training_load_weight),
                       map_location=device),
            strict=False
        )
        print("Model weights loaded.")

    # Initialize diffusion trainer
    trainer = GaussianDiffusionTrainer(
        net_model,
        beta_1=params.dif_beta_1,
        beta_T=params.dif_beta_T,
        T=params.dif_T
    ).to(device)

    # Prepare batch
    b = x.shape[0]
    x_0 = x.to(device)                # (B, 1152)
    c = c.to(device)                 # (B, 14)
    labels = labels.to(device).long()  # (B,)


    # Forward diffusion loss
    loss = trainer(x_0, c, labels)    # expect trainer to handle c (condition vector)
    loss = loss.sum() / b ** 2        # normalize

    return loss
