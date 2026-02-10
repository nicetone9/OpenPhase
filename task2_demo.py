"""
Task 2: Phase CVAE Training and Inference
Main training script for protein-RNA phase diagram prediction using Conditional Variational Autoencoder.
"""

import sys
import os
sys.path.append("./scripts")

import numpy as np
import torch
import swanlab
from loguru import logger
from tqdm.auto import trange

from db3 import DB3Dataset
from meta_config import create_meta_config
from model2 import PhaseCVAE, SelectedSystemDataset
from meta_plot import plot_grid_same_system, plot_roc_pr_curves

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def get_scaled_and_umap(obsm_data, data_name, save_normed=False):
    """
    Scale data and compute UMAP representation.
    
    Args:
        obsm_data: Dataset object with obsm attribute
        data_name: Name of the data key in obsm
        save_normed: Whether to save normalized data back to obsm
        
    Returns:
        UMAP 2D projection of scaled data
    """
    from umap import UMAP
    
    x_data_scaled = np.nan_to_num(StandardScaler().fit_transform(obsm_data.obsm[data_name]))
    
    umap_model = UMAP(n_components=2, random_state=42)
    umap_col = umap_model.fit_transform(x_data_scaled)
    
    if save_normed:
        obsm_data.obsm[f"{data_name}_normed"] = x_data_scaled
    
    return umap_col


def create_selected_dataset_and_loader(
    db3_ann,
    selected_indices,
    protein_embedding,
    dna_rna_embedding,
    condition_embedding="c_rnapsec_condition",
    train_test_split=0.5,
    batch_size=1024,
    random_state=42,
):
    """
    Create dataset and dataloader for selected protein-RNA system.
    
    Args:
        db3_ann: DB3 dataset
        selected_indices: Indices of selected samples
        protein_embedding: Key for protein embedding in obsm
        dna_rna_embedding: Key for DNA/RNA embedding in obsm
        condition_embedding: Key for condition data in obsm
        train_test_split: Test set proportion
        batch_size: Batch size for dataloaders
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader, grid_range_dict, phase_diag_dict
    """
    db3_selected = db3_ann[selected_indices]
    x_protein = db3_selected.obsm[protein_embedding]
    x_rna = db3_selected.obsm[dna_rna_embedding]
    y = db3_selected.obsm["y"]
    
    c_protein_rna = db3_selected.obsm[condition_embedding][:, -2:]
    c_protein_min, c_protein_max = np.min(c_protein_rna[:, 0]), np.max(c_protein_rna[:, 0])
    c_rna_min, c_rna_max = np.min(c_protein_rna[:, 1]), np.max(c_protein_rna[:, 1])
    
    # Split dataset
    train_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=train_test_split, random_state=random_state)
        .split(x_protein, y)
    )
    
    train_dataset = SelectedSystemDataset(x_protein, x_rna, c_protein_rna, y, train_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SelectedSystemDataset(x_protein, x_rna, c_protein_rna, y, test_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Phase diagram data
    c_protein_rna_y = np.c_[c_protein_rna, y]
    train_phase_diagram = c_protein_rna_y[train_idx]
    test_phase_diagram = c_protein_rna_y[test_idx]
    
    grid_range_dict = {
        "c_protein_min": c_protein_min,
        "c_protein_max": c_protein_max,
        "c_rna_min": c_rna_min,
        "c_rna_max": c_rna_max,
    }
    
    phase_diag_dict = {"train": train_phase_diagram, "test": test_phase_diagram}
    
    return train_loader, test_loader, grid_range_dict, phase_diag_dict


def create_condition_grid(grid_range_dict, num_points=30, margin=0.1):
    """
    Create a 2D grid of condition points for phase diagram visualization.
    
    Args:
        grid_range_dict: Dictionary with min/max values for protein and RNA concentrations
        num_points: Number of points along each axis
        margin: Margin to extend beyond min/max values (as fraction of range)
        
    Returns:
        Grid points as (n_points, 2) numpy array
    """
    c_protein_min = grid_range_dict["c_protein_min"]
    c_protein_max = grid_range_dict["c_protein_max"]
    c_rna_min = grid_range_dict["c_rna_min"]
    c_rna_max = grid_range_dict["c_rna_max"]
    
    protein_range = c_protein_max - c_protein_min
    rna_range = c_rna_max - c_rna_min
    
    c_protein_min_ext = c_protein_min - margin * protein_range
    c_protein_max_ext = c_protein_max + margin * protein_range
    c_rna_min_ext = c_rna_min - margin * rna_range
    c_rna_max_ext = c_rna_max + margin * rna_range
    
    c_protein_range = np.linspace(c_protein_min_ext, c_protein_max_ext, num_points)
    c_rna_range = np.linspace(c_rna_min_ext, c_rna_max_ext, num_points)
    c_protein_grid, c_rna_grid = np.meshgrid(c_protein_range, c_rna_range)
    grid_points = np.stack([c_protein_grid.ravel(), c_rna_grid.ravel()], axis=1)
    
    return grid_points


def grid_inference_same_system(model, grid_points, x_protein, x_rna, y_grid, device):
    """
    Perform inference on grid points to generate phase diagram predictions.
    
    Args:
        model: Trained PhaseCVAE model
        grid_points: Grid conditions (n_points, 2)
        x_protein: Protein embedding (1, protein_dim) or (protein_dim,)
        x_rna: RNA embedding (1, nucleic_dim) or (nucleic_dim,)
        y_grid: Grid labels (n_points,)
        device: PyTorch device
        
    Returns:
        Dictionary with predictions and losses for all grid points
    """
    phase_predictions = np.zeros(len(grid_points))
    reconstruction_errors = np.zeros(len(grid_points))
    kl_losses = np.zeros(len(grid_points))
    phase_losses = np.zeros(len(grid_points))
    total_losses = np.zeros(len(grid_points))
    
    model.eval()
    with torch.no_grad():
        for i, (grid_point, y_point) in enumerate(zip(grid_points, y_grid)):
            c_single = torch.tensor(grid_point, dtype=torch.float32).unsqueeze(0).to(device)
            
            if x_protein.dim() == 1:
                x_protein_single = x_protein.unsqueeze(0).to(device)
            else:
                x_protein_single = x_protein.to(device)
            
            if x_rna.dim() == 1:
                x_rna_single = x_rna.unsqueeze(0).to(device)
            else:
                x_rna_single = x_rna.to(device)
            
            y_single = torch.tensor(y_point, dtype=torch.float32).unsqueeze(0).to(device)
            
            loss_dict, y_pred = model.compute_loss(x_protein_single, x_rna_single, c_single, y_single)
            
            phase_predictions[i] = y_pred.cpu().numpy().flatten()[0]
            reconstruction_errors[i] = loss_dict["recon_loss"].item()
            kl_losses[i] = loss_dict["kl_loss"].item()
            phase_losses[i] = loss_dict["phase_loss"].item()
            total_losses[i] = loss_dict["total_loss"].item()
    
    return {
        "phase_predictions": phase_predictions,
        "reconstruction_errors": reconstruction_errors,
        "kl_losses": kl_losses,
        "phase_losses": phase_losses,
        "total_losses": total_losses,
    }


def grid_label_knn_inference(grid_points, reference_points, reference_labels, k=5):
    """
    Assign labels to grid points using k-nearest neighbors.
    
    Args:
        grid_points: Grid points to label (n_grid, 2)
        reference_points: Reference points with known labels (n_ref, 2)
        reference_labels: Labels for reference points (n_ref,)
        k: Number of nearest neighbors to consider
        
    Returns:
        Grid labels from majority voting (n_grid,)
    """
    knn = NearestNeighbors(n_neighbors=min(k, len(reference_points)))
    knn.fit(reference_points)
    distances, indices = knn.kneighbors(grid_points)
    
    grid_labels = np.zeros(len(grid_points))
    for i, neighbor_indices in enumerate(indices):
        neighbor_labels = reference_labels[neighbor_indices]
        grid_labels[i] = np.round(np.mean(neighbor_labels))
    
    return grid_labels


def select_by_protein_nucleic_acid(rnapsec_df, protein_id=None, nucleic_acid=None):
    """
    Select samples by protein ID and/or nucleic acid type.
    
    Args:
        rnapsec_df: DataFrame with protein and nucleic acid information
        protein_id: Protein ID to filter (None for no filter)
        nucleic_acid: Nucleic acid type to filter (None for no filter)
        
    Returns:
        Selected DataFrame and indices
    """
    if protein_id is None and nucleic_acid is None:
        selected_df = rnapsec_df
    elif protein_id is None:
        selected_df = rnapsec_df[rnapsec_df["Nucleic acid"] == nucleic_acid]
    elif nucleic_acid is None:
        selected_df = rnapsec_df[rnapsec_df["Protein ID"] == protein_id]
    else:
        selected_df = rnapsec_df[
            (rnapsec_df["Protein ID"] == protein_id) & (rnapsec_df["Nucleic acid"] == nucleic_acid)
        ]
    
    selected_indices = selected_df.index.astype(int).values
    return selected_df, selected_indices


def train_phase_cvae(
    train_loader,
    test_loader,
    grid_points,
    grid_range_dict,
    phase_diag_dict,
    y_grid_knn,
    x_protein_sample,
    x_rna_sample,
    num_epochs=100,
    learning_rate=1e-2,
    alpha=10.0,
    beta=0.1,
    gamma=5.0,
    save_dir="./task2/models",
    experiment_name="phase_cvae",
):
    """
    Train the PhaseCVAE model.
    
    Args:
        train_loader: Training dataloader
        test_loader: Testing dataloader
        grid_points: Grid points for phase diagram inference
        grid_range_dict: Grid range configuration
        phase_diag_dict: Phase diagram data (train/test splits)
        y_grid_knn: KNN-inferred labels for grid points
        x_protein_sample: Sample protein embedding for grid inference
        x_rna_sample: Sample RNA embedding for grid inference
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        alpha: Reconstruction loss weight
        beta: KL divergence loss weight
        gamma: Phase prediction loss weight
        save_dir: Directory to save models
        experiment_name: Experiment name for logging
        
    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data dimensions from first batch
    x_protein, x_rna, c, y = next(iter(train_loader))
    protein_dim = x_protein.shape[1]
    nucleic_dim = x_rna.shape[1]
    condition_dim = c.shape[1]
    
    model = PhaseCVAE(
        protein_dim=protein_dim,
        nucleic_dim=nucleic_dim,
        condition_dim=condition_dim,
        latent_dim=64,
        hidden_dim=128,
    ).to(device)
    
    batch_size = train_loader.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    swanlab.init(
        project="condition_phase",
        experiment_name=experiment_name,
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "protein_dim": protein_dim,
            "nucleic_dim": nucleic_dim,
            "condition_dim": condition_dim,
            "latent_dim": 64,
            "hidden_dim": 128,
        },
    )
    
    logger.info("=== Starting Training ===")
    best_test_loss = float("inf")
    
    for epoch in trange(num_epochs, desc="Training"):
        # Training phase
        model.train()
        train_losses = {"total": [], "recon": [], "kl": [], "phase": []}
        train_predictions, train_targets = [], []
        
        for x_protein_batch, x_rna_batch, c_batch, y_batch in train_loader:
            x_protein_batch = x_protein_batch.to(device)
            x_rna_batch = x_rna_batch.to(device)
            c_batch = c_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            loss_dict, y_pred = model.compute_loss(
                x_protein_batch, x_rna_batch, c_batch, y_batch,
                alpha=alpha, beta=beta, gamma=gamma
            )
            
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses["total"].append(loss.item())
            train_losses["recon"].append(loss_dict["recon_loss"].item())
            train_losses["kl"].append(loss_dict["kl_loss"].item())
            train_losses["phase"].append(loss_dict["phase_loss"].item())
            
            train_predictions.extend(y_pred.detach().cpu().numpy().squeeze())
            train_targets.extend(y_batch.cpu().numpy().squeeze())
        
        # Evaluation phase
        model.eval()
        test_losses = {"total": [], "recon": [], "kl": [], "phase": []}
        test_predictions, test_targets = [], []
        
        with torch.no_grad():
            for x_protein_batch, x_rna_batch, c_batch, y_batch in test_loader:
                x_protein_batch = x_protein_batch.to(device)
                x_rna_batch = x_rna_batch.to(device)
                c_batch = c_batch.to(device)
                y_batch = y_batch.to(device)
                
                loss_dict, y_pred = model.compute_loss(
                    x_protein_batch, x_rna_batch, c_batch, y_batch,
                    alpha=alpha, beta=beta, gamma=gamma
                )
                
                test_losses["total"].append(loss_dict["total_loss"].item())
                test_losses["recon"].append(loss_dict["recon_loss"].item())
                test_losses["kl"].append(loss_dict["kl_loss"].item())
                test_losses["phase"].append(loss_dict["phase_loss"].item())
                
                test_predictions.extend(y_pred.cpu().numpy().squeeze())
                test_targets.extend(y_batch.cpu().numpy().squeeze())
        
        # Calculate metrics
        train_pred_binary = [1 if x > 0.5 else 0 for x in train_predictions]
        test_pred_binary = [1 if x > 0.5 else 0 for x in test_predictions]
        
        train_accuracy = accuracy_score(train_targets, train_pred_binary)
        test_accuracy = accuracy_score(test_targets, test_pred_binary)
        train_auc = roc_auc_score(train_targets, train_predictions)
        test_auc = roc_auc_score(test_targets, test_predictions)
        
        # Generate visualizations
        train_roc_pr_img = plot_roc_pr_curves(train_targets, train_predictions)
        test_roc_pr_img = plot_roc_pr_curves(test_targets, test_predictions)
        
        grid_inference_dict = grid_inference_same_system(
            model, grid_points, x_protein_sample, x_rna_sample, y_grid_knn, device
        )
        grid_inference_img = plot_grid_same_system(grid_points, grid_inference_dict, phase_diag_dict)
        
        # Log to SwanLab
        swanlab.log(
            {
                "epoch": epoch,
                "img/grid_phase_diagram": grid_inference_img,
                "img/train_roc_pr_curve": train_roc_pr_img,
                "img/test_roc_pr_curve": test_roc_pr_img,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train/total_loss": np.mean(train_losses["total"]),
                "train/recon_loss": np.mean(train_losses["recon"]),
                "train/kl_loss": np.mean(train_losses["kl"]),
                "train/phase_loss": np.mean(train_losses["phase"]),
                "train/accuracy": train_accuracy,
                "train/auc": train_auc,
                "test/total_loss": np.mean(test_losses["total"]),
                "test/recon_loss": np.mean(test_losses["recon"]),
                "test/kl_loss": np.mean(test_losses["kl"]),
                "test/phase_loss": np.mean(test_losses["phase"]),
                "test/accuracy": test_accuracy,
                "test/auc": test_auc,
            },
            print_to_console=False,
        )
        
        # Learning rate scheduling
        scheduler.step(np.mean(test_losses["total"]))
        
        # Save best model
        if np.mean(test_losses["total"]) < best_test_loss:
            best_test_loss = np.mean(test_losses["total"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_dir, f"phase_cvae_epoch{epoch}_loss{best_test_loss:.4f}.pth"),
            )
            logger.info(
                f"Saved best model at epoch {epoch} with test loss: {best_test_loss:.4f}"
            )
    
    swanlab.finish()
    return model


if __name__ == "__main__":
    logger.info("=== Task 2: Phase CVAE Training ===")
    
    # Load dataset
    meta_config = create_meta_config(
        protein="mtdp", dna_rna="dictionary", condition="rnapsec_condition"
    )
    db3 = DB3Dataset(root="../data", name="DB3", embedding_config=meta_config, download=True)
    db3_ann = db3[:]
    db3_df = db3_ann.to_df()
    
    # Select system
    selected_df, selected_indices = select_by_protein_nucleic_acid(
        db3_df, protein_id="P35637", nucleic_acid="MAPT"
    )
    
    logger.info(f"Selected system: {len(selected_indices)} samples")
    logger.info(f"Protein-RNA combinations:\n{selected_df[['Protein ID', 'Nucleic acid']].value_counts()}")
    
    # Prepare data
    train_loader, test_loader, grid_range_dict, phase_diag_dict = (
        create_selected_dataset_and_loader(
            db3_ann,
            selected_indices,
            protein_embedding="x_esmc_normed",
            dna_rna_embedding="x_rna_rnapsec_rna_normed",
        )
    )
    
    # Create grid
    grid_points = create_condition_grid(grid_range_dict, num_points=30, margin=0.25)
    reference_points = np.vstack(
        (phase_diag_dict["train"][:, :2], phase_diag_dict["test"][:, :2])
    )
    reference_labels = np.hstack(
        (phase_diag_dict["train"][:, 2], phase_diag_dict["test"][:, 2])
    )
    y_grid_knn = grid_label_knn_inference(grid_points, reference_points, reference_labels, k=5)
    
    # Get sample embeddings for grid inference
    x_protein_sample, x_rna_sample, _, _ = next(iter(train_loader))
    
    # Train model
    model = train_phase_cvae(
        train_loader,
        test_loader,
        grid_points,
        grid_range_dict,
        phase_diag_dict,
        y_grid_knn,
        x_protein_sample[0],
        x_rna_sample[0],
        num_epochs=100,
        learning_rate=1e-2,
        alpha=10.0,
        beta=0.1,
        gamma=5.0,
        experiment_name="task2_phase_cvae_db3",
    )
    
    logger.info("=== Training Complete ===")
