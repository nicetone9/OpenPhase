import os, argparse, torch, wandb
from torch import optim
from tqdm import tqdm

import sys
#sys.path.append("C:/Users/USER/Documents/model")

from JTAE.jtae_ import jtae
from utils.scheduler import GradualWarmupScheduler
from utils.losses import loss_function
from data.loader_train_val import get_train_val_loaders

def save_checkpoint(epoch, model, optimizer, scheduler, args, save_dir):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
    }
    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    print(f"[INFO] Loading checkpoint → {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'] + 1


def validate(model, val_loader, args):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(args.device)
            c = batch['c'].to(args.device)
            y = batch['y'].to(args.device)

            # model must return (x_hat, z_rep, diff_loss)
            x_hat, z_rep, diff_loss = model(x, c, y)

            loss = loss_function(args, x_hat, y, z_rep, diff_loss)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(args):
    save_dir = os.path.join("checkpoints", args.model)
    os.makedirs(save_dir, exist_ok=True)

    # ----- W&B -----
    if args.use_wandb:
        wandb.init(
            entity="adrita78-carnegie-mellon-university",
            project="Open-Phase",
            config=vars(args)
        )

    # ----- Data -----
    train_loader, val_loader = get_train_val_loaders(
        batch_size=args.batch_size,
        data_path=args.data_path,
        val_split=0.1,
        seed=42
    )

    # ----- Model -----
    model = jtae(hparams=args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=args.multiplier,
        warm_epoch=args.n_epochs // 10,
        after_scheduler=cosine
    )

    # ----- Resume -----
    start_epoch = 0
    if args.resume_path and os.path.exists(args.resume_path):
        start_epoch = load_checkpoint(args.resume_path, model, optimizer, warmup, args.device)

    # ----- Training -----
    print("Training Started.")
    for epoch in range(start_epoch, args.n_epochs):

        model.train()
        running_loss = 0.0

        with tqdm(train_loader, dynamic_ncols=True) as t:
            for batch in t:

                x = batch['x'].to(args.device)
                c = batch['c'].to(args.device)
                y = batch['y'].to(args.device)

                optimizer.zero_grad()

                # Model forward pass
                x_hat, z_rep, diff_loss = model(x, c, y)

                # Compute loss
                loss = loss_function(args, x_hat, x, z_rep, diff_loss)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_description(f"Epoch [{epoch+1}/{args.n_epochs}] Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, args)

        warmup.step()

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, step=epoch)

        # Save every 50 epochs + last epoch
        if (epoch + 1) % 50 == 0 or epoch == args.n_epochs - 1:
            save_checkpoint(epoch, model, optimizer, warmup, args, save_dir)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="jtae")
    parser.add_argument("--data_path", type=str, default="/home/peiranj/model/data/protein(1).no_missing.h5ad")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--multiplier", type=float, default=2.5)
    parser.add_argument("--resume_path", type=str, default=None)

    # Model hyperparameters
    parser.add_argument("--input_dim", type=int, default=1152)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--eta_val", type=float, default=0.001)
    parser.add_argument("--gamma_val", type=float, default=1.0)
    parser.add_argument("--sigma_val", type=float, default=1.5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--probs", type=float, default=0.2)
    parser.add_argument("--kernel_size", type=int, default=3)

    # Diffusion hyperparameters
    parser.add_argument("--dif_T", type=int, default=1000)
    parser.add_argument("--dif_channel", type=int, default=64)
    parser.add_argument("--dif_channel_mult", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--dif_res_blocks", type=int, default=2)
    parser.add_argument("--dif_dropout", type=float, default=0.15)
    parser.add_argument("--dif_beta_1", type=float, default=1e-4)
    parser.add_argument("--dif_beta_T", type=float, default=0.028)

    args = parser.parse_args()

    print(f"[INFO] Using device: {args.device}")
    train(args)
