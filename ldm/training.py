import os
import numpy as np
import torch
from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer

def train(params, batch, net_model):
    device = params.device

    # Extract inputs
    x_0 = batch["x"].to(device)         # Input 
    labels = batch["y"].to(device)      # Binary labels
    cond = batch["c"].to(device)        # Conditional vector 

    if params.dif_training_load_weight is not None:
        net_model.load_state_dict(
            torch.load(os.path.join(params.dif_sav_dir, params.dif_training_load_weight), map_location=device),
            strict=False
        )
        print("Model weight loaded.")

    # Initialize diffusion trainer
    trainer = GaussianDiffusionTrainer(
        net_model, params.dif_beta_1, params.dif_beta_T, params.dif_T
    ).to(device)

    # Possibly apply classifier-free guidance masking (10% of the time)
    if np.random.rand() < 0.1:
        labels = torch.zeros_like(labels)

    labels = labels.long()  # In case they're float

    # Compute loss
    loss = trainer(x_0, labels, cond)
    loss = loss.sum() / (x_0.shape[0] ** 2)

    return loss
