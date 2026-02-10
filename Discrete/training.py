import torch
import torch.nn.functional as F
import os
import copy
import numpy as np
import datetime
import anndata as ad
from tqdm import tqdm
import torch.optim as optim
from utils_ import inf_iterator, enable_dropout, cal_stats_metric
from torch.utils.data import DataLoader
from dataprocessing_loading import get_loader
from sklearn.model_selection import train_test_split
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
import esm
from protein_diffusion_model import ProteinDiffusionModel
from protein_diffusion_model import UNet
from pathlib import Path
from omegaconf import OmegaConf
from prettytable import PrettyTable
from evaluator import Evaluator
from torch.optim import Adam, lr_scheduler

class Trainer(object):
    def __init__(
        self,
        config,
        diffusion_model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        device,
        output_dir,
        scheduler=None,
        train_batch_size=512,
        train_num_steps=200000,
        save_and_sample_every=100,
        ensemble_num=50,
        ddim_steps=50,
        sample_method="ddim",
        experiment=None,
    ):
        super().__init__()

        self.device = device
        self.model = diffusion_model.to(device)
        self.best_model = None

        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment = experiment

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_iterator = inf_iterator(train_dataloader)

        self.iter_one_epoch = len(train_dataloader)
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every

        self.ensemble_num = ensemble_num
        self.ddim_steps = ddim_steps
        self.sample_method = sample_method

        self.evaluator = Evaluator()

        self.step = 0
        self.epoch = 0

        self.best_val_recovery = -1.0
        self.best_val_epoch = 0
        self.best_val_step = 0
        self.best_val_perplexity = float("inf")

        self.results_folder = output_dir
        Path(self.results_folder + "/model").mkdir(parents=True, exist_ok=True)

        self.train_table = PrettyTable(["Epoch", "Step", "Train Loss"])
        self.val_table = PrettyTable(["Epoch", "Step", "Median Recovery", "Perplexity"])
        self.test_table = PrettyTable(["Epoch", "Step", "Median Recovery", "Perplexity"])

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------
    def train(self):
        print(f"Starting training for {self.train_num_steps} steps")
        epoch_loss = 0.0

        with tqdm(total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                x, c, y = next(self.train_iterator)
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                y = y.to(self.device) if y is not None else None

                self.optimizer.zero_grad()

                base_loss, mask_loss = self.model(x, c=c, y=y)
                base_loss = base_loss or torch.tensor(0.0, device=self.device)
                mask_loss = mask_loss or torch.tensor(0.0, device=self.device)

                loss = base_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.step += 1
                epoch_loss += loss.item()
                pbar.update(1)

                if self.experiment:
                    self.experiment.log_metric("train_loss", loss.item(), step=self.step)

                # ---- epoch bookkeeping ----
                if self.step % self.iter_one_epoch == 0:
                    self.epoch += 1
                    self.train_table.add_row(
                        [self.epoch, self.step, epoch_loss / self.iter_one_epoch]
                    )
                    epoch_loss = 0.0

                # ---- validation ----
                if self.step % (self.save_and_sample_every * self.iter_one_epoch) == 0:
                    self.validate()

        print("Training complete")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        enable_dropout(self.model)

        recoveries = []
        all_logits, all_targets = [], []

        with torch.no_grad():
            for x, c, y in tqdm(self.val_dataloader, desc="Validation"):
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                B, L = x.shape

                ens_logits = []
                for _ in range(self.ensemble_num):
                    logits, _, _ = self.model.mc_ddim_sample(
                        x=x, cond=c, diverse=True, step=self.ddim_steps
                    )
                    ens_logits.append(logits)

                mean_logits = torch.stack(ens_logits).mean(0).cpu()
                all_logits.append(mean_logits)
                all_targets.append(x.cpu())

                for i in range(B):
                    pred = mean_logits[i].argmax(dim=-1)
                    tgt = x[i].cpu()
                    recoveries.append(self.evaluator.cal_recovery(pred, tgt))

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        perplexity = self.evaluator.cal_perplexity(
            all_logits.view(-1, all_logits.size(-1)),
            all_targets.view(-1),
        )

        mean_rec, median_rec = cal_stats_metric(recoveries)

        self.val_table.add_row([self.epoch, self.step, median_rec, perplexity])

        if median_rec > self.best_val_recovery:
            self.best_model = copy.deepcopy(self.model)
            self.best_val_recovery = median_rec
            self.best_val_perplexity = perplexity
            self.best_val_epoch = self.epoch
            self.best_val_step = self.step

        print(f"[VAL] median recovery={median_rec:.4f}, ppl={perplexity:.4f}")

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    def test(self, epoch):
        self.best_model.eval()

        recoveries = []
        n45, n62, n80, n90 = [], [], [], []

        all_logits, all_targets = [], []

        with torch.no_grad():
            for x, c, _ in tqdm(self.test_dataloader, desc="Test"):
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                B, L = x.shape

                ens_logits = []
                for _ in range(self.ensemble_num):
                    logits, _, _ = self.best_model.mc_ddim_sample(
                        x=x, cond=c, diverse=False
                    )
                    ens_logits.append(logits)

                mean_logits = torch.stack(ens_logits).mean(0).cpu()
                all_logits.append(mean_logits)
                all_targets.append(x.cpu())

                for i in range(B):
                    pred = mean_logits[i].argmax(dim=-1)
                    tgt = x[i].cpu()
                    recoveries.append(self.evaluator.cal_recovery(pred, tgt))
                    a, b, c_, d = self.evaluator.cal_all_blosum_nssr(pred, tgt)
                    n45.append(a)
                    n62.append(b)
                    n80.append(c_)
                    n90.append(d)

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        perplexity = self.evaluator.cal_perplexity(
            all_logits.view(-1, all_logits.size(-1)),
            all_targets.view(-1),
        )

        mean_rec, median_rec = cal_stats_metric(recoveries)

        self.test_table.add_row([epoch, self.best_val_step, median_rec, perplexity])

        print(f"[TEST] median recovery={median_rec:.4f}, ppl={perplexity:.4f}")

        return {
            "median_recovery": median_rec,
            "mean_recovery": mean_rec,
            "perplexity": perplexity,
            "mean_nssr62": np.mean(n62),
        }

        
import torch
import torch.nn.functional as F
import os
import copy
import numpy as np
import datetime
import anndata as ad
from tqdm import tqdm
import torch.optim as optim
from utils_ import inf_iterator, enable_dropout, cal_stats_metric
from torch.utils.data import DataLoader
from dataprocessing_loading import get_loader
from sklearn.model_selection import train_test_split
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
import esm
from protein_diffusion_model import ProteinDiffusionModel
from protein_diffusion_model import UNet
from pathlib import Path
from omegaconf import OmegaConf
from prettytable import PrettyTable
from evaluator import Evaluator
from torch.optim import Adam, lr_scheduler




class Trainer(object):
    def __init__(
        self,
        config,
        diffusion_model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        device,
        output_dir,
        scheduler=None,
        train_batch_size=512,
        train_num_steps=200000,
        save_and_sample_every=100,
        ensemble_num=50,
        ddim_steps=50,
        sample_method="ddim",
        experiment=None,
    ):
        super().__init__()

        self.device = device
        self.model = diffusion_model.to(device)
        self.best_model = None

        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment = experiment

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_iterator = inf_iterator(train_dataloader)

        self.iter_one_epoch = len(train_dataloader)
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every

        self.ensemble_num = ensemble_num
        self.ddim_steps = ddim_steps
        self.sample_method = sample_method

        self.evaluator = Evaluator()

        self.step = 0
        self.epoch = 0

        self.best_val_recovery = -1.0
        self.best_val_epoch = 0
        self.best_val_step = 0
        self.best_val_perplexity = float("inf")

        self.results_folder = output_dir
        Path(self.results_folder + "/model").mkdir(parents=True, exist_ok=True)

        self.train_table = PrettyTable(["Epoch", "Step", "Train Loss"])
        self.val_table = PrettyTable(["Epoch", "Step", "Median Recovery", "Perplexity"])
        self.test_table = PrettyTable(["Epoch", "Step", "Median Recovery", "Perplexity"])

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------
    def train(self):
        print(f"Starting training for {self.train_num_steps} steps")
        epoch_loss = 0.0

        with tqdm(total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                x, c, y = next(self.train_iterator)
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                y = y.to(self.device) if y is not None else None

                self.optimizer.zero_grad()

                base_loss, mask_loss = self.model(x, c=c, y=y)
                base_loss = base_loss or torch.tensor(0.0, device=self.device)
                mask_loss = mask_loss or torch.tensor(0.0, device=self.device)

                loss = base_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.step += 1
                epoch_loss += loss.item()
                pbar.update(1)

                if self.experiment:
                    self.experiment.log_metric("train_loss", loss.item(), step=self.step)

                # ---- epoch bookkeeping ----
                if self.step % self.iter_one_epoch == 0:
                    self.epoch += 1
                    self.train_table.add_row(
                        [self.epoch, self.step, epoch_loss / self.iter_one_epoch]
                    )
                    epoch_loss = 0.0

                # ---- validation ----
                if self.step % (self.save_and_sample_every * self.iter_one_epoch) == 0:
                    self.validate()

        print("Training complete")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        enable_dropout(self.model)

        recoveries = []
        all_logits, all_targets = [], []

        with torch.no_grad():
            for x, c, y in tqdm(self.val_dataloader, desc="Validation"):
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                B, L = x.shape

                ens_logits = []
                for _ in range(self.ensemble_num):
                    logits, _, _ = self.model.mc_ddim_sample(
                        x=x, cond=c, diverse=True, step=self.ddim_steps
                    )
                    ens_logits.append(logits)

                mean_logits = torch.stack(ens_logits).mean(0).cpu()
                all_logits.append(mean_logits)
                all_targets.append(x.cpu())

                for i in range(B):
                    pred = mean_logits[i].argmax(dim=-1)
                    tgt = x[i].cpu()
                    # replace pad token if necessary
                    pred_safe = pred.clone()
                    pred_safe[pred_safe == 20] = 0
                    recoveries.append(self.evaluator.cal_recovery(pred_safe, tgt))

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        perplexity = self.evaluator.cal_perplexity(
            all_logits.view(-1, all_logits.size(-1)),
            all_targets.view(-1),
        )

        mean_rec, median_rec = cal_stats_metric(recoveries)

        self.val_table.add_row([self.epoch, self.step, median_rec, perplexity])

        if median_rec > self.best_val_recovery:
            self.best_model = copy.deepcopy(self.model)
            self.best_val_recovery = median_rec
            self.best_val_perplexity = perplexity
            self.best_val_epoch = self.epoch
            self.best_val_step = self.step

        print(f"[VAL] median recovery={median_rec:.4f}, ppl={perplexity:.4f}")

    # ------------------------------------------------------------------
    # TEST (FIXED: handles pad token and saves sequences)
    # ------------------------------------------------------------------
    def test(self, epoch):
        self.best_model.eval()

        recoveries = []
        n45, n62, n80, n90 = [], [], [], []

        all_logits, all_targets = [], []
        all_gen_seqs = []  # save generated sequences

        with torch.no_grad():
            for x, c, _ in tqdm(self.test_dataloader, desc="Test"):
                x = x.to(self.device)
                c = c.to(self.device) if c is not None else None
                B, L = x.shape

                ens_logits = []
                seqs = None

                # ----- Ensemble sampling -----
                for _ in range(self.ensemble_num):
                    logits, _, seqs = self.best_model.mc_ddim_sample(
                        x=x, cond=c, diverse=False
                    )
                    ens_logits.append(logits)

                if seqs is not None:
                    all_gen_seqs.extend(seqs)

                mean_logits = torch.stack(ens_logits).mean(0).cpu()
                all_logits.append(mean_logits)
                all_targets.append(x.cpu())

                # ----- Evaluation per sequence -----
                for i in range(B):
                    pred = mean_logits[i].argmax(dim=-1)
                    tgt = x[i].cpu()

                    # ----- FIX: replace pad token (20) with valid AA index (0) -----
                    #pred_safe = pred.clone()
                    #pred_safe[pred_safe == 20] = 0
                    pred_safe = pred.clone().clamp(0, 19)
                    tgt_safe = tgt.clone().clamp(0, 19)

                    recoveries.append(self.evaluator.cal_recovery(pred_safe, tgt))
                    #a, b, c_, d = self.evaluator.cal_all_blosum_nssr(pred_safe, tgt)
                    #n45.append(a)
                    #n62.append(b)
                    #n80.append(c_)
                    #n90.append(d)

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        perplexity = self.evaluator.cal_perplexity(
            all_logits.view(-1, all_logits.size(-1)),
            all_targets.view(-1),
        )

        mean_rec, median_rec = cal_stats_metric(recoveries)
        self.test_table.add_row([epoch, self.best_val_step, median_rec, perplexity])

        # ---------------- SAVE GENERATED SEQUENCES ----------------
        save_path = os.path.join(
            self.results_folder, f"test_generated_epoch_{epoch}.fasta"
        )
        with open(save_path, "w") as f:
            for i, seq in enumerate(all_gen_seqs):
                f.write(f">gen_{i}\n{seq}\n")

        print(f"[TEST] Saved {len(all_gen_seqs)} sequences to {save_path}")
        print(f"[TEST] median recovery={median_rec:.4f}, ppl={perplexity:.4f}")

        return {
            "median_recovery": median_rec,
            "mean_recovery": mean_rec,
            "perplexity": perplexity,
            #"mean_nssr62": np.mean(n62),
        }
        
def build_config():
    return OmegaConf.create({
        "experiment": {
            "name": "protein_diffusion_run1",
        },
        "training": {
            "batch_size": 8,
            "num_steps": 21,300,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "use_scheduler": True,
        }
    })



def build_diffusion_model(cfg, device):
    model = UNet(
        T=1000,
        ch=64,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        dropout=0.1,
        cond_dim=14,
        num_classes=21
    )

    prior_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    diffusion_model = ProteinDiffusionModel(
        model=model,
        prior_model=prior_model,
        esm_batch_converter=batch_converter,
        timesteps=100,
        loss_type="CE",
        objective="pred_x0",
        noise_type="marginal",
        marginal_dist_path="/home/peiranj/mapdiff/train_marginal_x.pt"
    )

    return diffusion_model.to(device)



def build_optimizer(cfg, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.95, 0.999),
        weight_decay=cfg.training.weight_decay
    )

    scheduler = None
    if cfg.training.use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.training.lr,
            total_steps=cfg.training.num_steps
        )

    return optimizer, scheduler


def build_dataloaders(cfg):
    adata = ad.read_h5ad(
        "/home/peiranj/mapdiff/protein(1).no_missing (2).h5ad"
    )

    x = adata.obsm["x_seq2onehot"]   # (N, L, 21)
    c = adata.obsm["c_droppler"]     # (N, D_c)
    y = adata.obsm["y"]

    x_trainval, x_test, c_trainval, c_test, y_trainval, y_test = train_test_split(
        x, c, y, test_size=0.1, random_state=42
    )

    x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(
        x_trainval, c_trainval, y_trainval, test_size=0.1, random_state=42
    )

    train_loader = get_loader(
        x_train, c_train, y_train,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )

    val_loader = get_loader(
        x_val, c_val, y_val,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )

    test_loader = get_loader(
        x_test, c_test, y_test,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def build_trainer(cfg, model, loaders, optimizer, scheduler, device):
    train_loader, val_loader, test_loader = loaders

    return Trainer(
        config=cfg,
        diffusion_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir="/home/peiranj/mapdiff/Results/",
        train_batch_size=cfg.training.batch_size,
        train_num_steps=cfg.training.num_steps,
        save_and_sample_every=1,
        ddim_steps=500,
        sample_method="ddim"
    )


def load_and_test(trainer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    trainer.best_model = copy.deepcopy(trainer.model)
    trainer.best_model.load_state_dict(ckpt["model"])
    trainer.best_model.to(device)
    trainer.best_model.eval()

    trainer.test(epoch=ckpt["epoch"])



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = build_config()
    diffusion_model = build_diffusion_model(cfg, device)
    optimizer, scheduler = build_optimizer(cfg, diffusion_model)
    loaders = build_dataloaders(cfg)

    trainer = build_trainer(
        cfg, diffusion_model, loaders, optimizer, scheduler, device
    )

    load_and_test(
        trainer,
        ckpt_path="/home/peiranj/mapdiff/Results/model/"
                  "protein_diffusion_run1_epoch_50_epochs_10650_steps_2025-11-24_05-38-40.pt",
        device=device
    )


if __name__ == "__main__":
    main()





