import os
import pickle
import sys
import warnings

import swanlab
import numpy as np
import torch
from loguru import logger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, cohen_kappa_score, make_scorer, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


sys.path.append("./scripts")
sys.path.append("./droppler")
warnings.filterwarnings("ignore")

# %%
# with open('params_all.yaml', 'r') as f:
#     params = yaml.safe_load(f)


def load_rnapsec_dataset(system_dataset):
    *x, c, y = system_dataset.get_tensor()
    # Concatenate x along the last axis and convert x, c, y to numpy arrays
    x_concat = np.concatenate([np.array(arr) for arr in x], axis=-1)
    c = np.array(c)
    y = np.array(y)

    xc = np.concatenate([x_concat, c], axis=-1)
    scaler = StandardScaler()
    xc_norm = scaler.fit_transform(xc)

    return xc_norm, y


def objective_score(trial, X_train, y_train):
    scorer = make_scorer(roc_auc_score, multi_class="ovr", average="macro")
    max_depth = trial.suggest_categorical("max_depth", [5, 6, 7, 8, 9, 10, None])
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    learning_rate = trial.suggest_categorical("learning_rate", [0.5, 1.0, 1.5, 2.0, 2.5])
    model = AdaBoostClassifier(estimator=base_estimator, learning_rate=learning_rate, random_state=30)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)

    return np.mean(scores)


def set_best_params(best_params):
    base_estimator = DecisionTreeClassifier(max_depth=best_params["max_depth"])
    model = AdaBoostClassifier(estimator=base_estimator, learning_rate=best_params["learning_rate"], random_state=30)

    return model


def baseline_rnapsec(system_name, split_method, protein_method, dna_rna_method, condition_method):
    logger.info(
        f"Running baseline_rnapsec for system: {system_name}, split_method: {split_method}, "
        f"protein_method: {protein_method}, dna_rna_method: {dna_rna_method}, condition_method: {condition_method}"
    )
    import optuna
    from db1 import DB1Dataset
    from meta_config import create_meta_config

    system_meta_config = create_meta_config(protein=protein_method, dna_rna=dna_rna_method, condition=condition_method)
    system_dataset = DB1Dataset(root="./data", name="DB1", system=system_name, embedding_config=system_meta_config)

    xc_norm, y = load_rnapsec_dataset(system_dataset)
    logger.info(f"Loaded dataset with shapes: {xc_norm.shape}, {y.shape}")
    logger.info(f"xc_norm mean: {np.mean(xc_norm):.4f}, std: {np.std(xc_norm):.4f}")

    train_idx, test_idx = system_dataset.get_idx_split(split_method=split_method)
    X_train, X_test = xc_norm[train_idx], xc_norm[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info(f"Split dataset into train {X_train.shape} and test {X_test.shape} by {split_method}")

    # optuna hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_score(trial, X_train, y_train), n_trials=32, n_jobs=32)
    best_params = study.best_trial.params
    clf = set_best_params(best_params)
    model_opt = clf.fit(X_train, y_train)

    test_preds = model_opt.predict(X_test)
    test_probs = model_opt.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")
    auprc = average_precision_score(y_test, test_probs, average="macro")
    kappa = cohen_kappa_score(y_test, test_preds)
    mcc = matthews_corrcoef(y_test, test_preds)

    swanlab.log({"system_name": system_name})
    swanlab.log({"split_method": split_method})
    swanlab.log({"protein_embed_method": protein_method})
    swanlab.log({"dna_rna_embed_method": dna_rna_method})
    swanlab.log({"condition_embed_method": condition_method})
    swanlab.log({"auroc": auroc})
    swanlab.log({"auprc": auprc})
    swanlab.log({"kappa": kappa})
    swanlab.log({"mcc": mcc})


def load_droppler_dataset(system_dataset):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def getEmbeddingValues(seq):
        "Convert amino acid sequence to ind, copied from droppler/utils.py"
        r = []
        listAA = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        for i in seq:
            if i not in listAA:
                r.append(0)
            else:
                r.append(listAA.index(i) + 1)
        return r

    assert system_dataset.embedding_config.condition_embedding == "droppler", "Condition method must be 'droppler' for this dataset"

    seqs = system_dataset[:].to_df()["Protein sequences"]
    seqs_embd = [torch.tensor(getEmbeddingValues(seq)) for seq in seqs]
    seqs_lens = [len(seq) for seq in seqs]
    X = pad_sequence(seqs_embd, padding_value=0, batch_first=True)
    *_, c, y = system_dataset.get_tensor()

    c = np.array(c)
    y = np.array(y)

    temp = c[:, :10]
    conc = c[:, 10]
    salt = c[:, 11]
    crowd = c[:, 12]
    ph = c[:, 13]

    X_cond = np.c_[temp, conc, crowd, salt, ph]

    return X, X_cond, seqs_lens, y


def baseline_droppler(system_name, split_method, protein_method, dna_rna_method, condition_method):
    logger.info(
        f"Running baseline_droppler for system: {system_name}, split_method: {split_method}, "
        f"protein_method: {protein_method}, dna_rna_method: {dna_rna_method}, condition_method: {condition_method}"
    )
    import SA
    from db1 import DB1Dataset
    from meta_config import create_meta_config
    from NNWrappers import NNwrapper

    # 构建数据集和特征
    system_meta_config = create_meta_config(protein=protein_method, dna_rna=dna_rna_method, condition=condition_method)
    system_dataset = DB1Dataset(root="./data", name="DB1", system=system_name, embedding_config=system_meta_config)
    X, Xcond, lens, y = load_droppler_dataset(system_dataset)

    scaler = pickle.load(open("./droppler/marshalled/scaler.pickle", "rb"))
    scaler.clip = False
    Xcond = scaler.transform(Xcond)

    train_idx, test_idx = system_dataset.get_idx_split(split_method=split_method)
    X_train, X_test = X[train_idx], X[test_idx]
    Xcond_train, Xcond_test = Xcond[train_idx], Xcond[test_idx]
    lens_train, lens_test = np.array(lens)[train_idx], np.array(lens)[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = SA.selfatt_RRN(nfea=20, nCond=14, hidden_size=10, num_layers=2, heads=10, name="droppler_baseline").to(device)
    wrapper = NNwrapper(model)

    tmpx_train = [xa.tolist() for xa in Xcond_train]
    tmpx_test = [xa.tolist() for xa in Xcond_test]

    Xcond_train_tensor = [torch.tensor(x, dtype=torch.float32, device=device) for x in tmpx_train]
    Xcond_test_tensor = [torch.tensor(x, dtype=torch.float32, device=device) for x in tmpx_test]
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    # 训练参数
    epochs = 10
    batch_size = 1024
    weight_decay = 1e-4
    learning_rate = 3e-3

    wrapper.fit(
        X_train,
        Xcond_train_tensor,
        torch.tensor(y_train, dtype=torch.float32, device=device),
        lens_train,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        dev=device,
        save_model_every=100,
        LOG=True,
        model_save_dir="./task1/models",
    )

    model = torch.load("./task1/models/droppler_baseline.best.pt", map_location=device, weights_only=False)
    wrapper = NNwrapper(model)

    y_probs, _ = wrapper.predict(X_test, Xcond_test_tensor, lens_test, dev=device)
    y_probs = np.array(y_probs)

    auroc = roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro")
    auprc = average_precision_score(y_test, y_probs, average="macro")
    kappa = cohen_kappa_score(y_test, y_probs >= 0.5)
    mcc = matthews_corrcoef(y_test, y_probs >= 0.5)

    swanlab.log({"system_name": system_name})
    swanlab.log({"split_method": split_method})
    swanlab.log({"protein_embed_method": protein_method})
    swanlab.log({"dna_rna_embed_method": dna_rna_method})
    swanlab.log({"condition_embed_method": condition_method})
    swanlab.log({"epochs": epochs})
    swanlab.log({"batch_size": batch_size})
    swanlab.log({"weight_decay": weight_decay})
    swanlab.log({"learning_rate": learning_rate})
    swanlab.log({"auroc": auroc})
    swanlab.log({"auprc": auprc})
    swanlab.log({"kappa": kappa})
    swanlab.log({"mcc": mcc})
    logger.info(f"Test - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Kappa: {kappa:.4f}, MCC: {mcc:.4f}")


def load_system_dataset(system_dataset):
    """Load dataset and return features and labels."""
    if system_dataset.system == "protein(1)":
        # For protein(1) system, we use the DB1Dataset
        x, c, y = system_dataset.get_tensor()
        x = StandardScaler().fit_transform(x)
        c = StandardScaler().fit_transform(c)
    else:
        raise NotImplementedError(f"System {system_dataset.system} is not implemented yet.")

    return x, c, y


def protein1_condformer(system_name, split_method, protein_method, dna_rna_method, condition_method):
    logger.info(
        f"Running baseline_condformer for system: {system_name}, split_method: {split_method}, "
        f"protein_method: {protein_method}, dna_rna_method: {dna_rna_method}, condition_method: {condition_method}"
    )

    from db1 import DB1Dataset
    from meta_config import create_meta_config
    from model1 import PhaseClassificationHead, Protein1SystemEncoder

    assert system_name == "protein(1)", "This baseline only supports protein(1) system!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    system_meta_config = create_meta_config(protein=protein_method, dna_rna=dna_rna_method, condition=condition_method)
    system_dataset = DB1Dataset(root="./data", name="DB1", system=system_name, embedding_config=system_meta_config)

    x, c, y = load_system_dataset(system_dataset)

    train_idx, test_idx = system_dataset.get_idx_split(split_method=split_method)
    x_train, x_test = x[train_idx], x[test_idx]
    c_train, c_test = c[train_idx], c[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    system_encoder = Protein1SystemEncoder(
        protein_dim=29 if protein_method == "rnapsec_protein" else 1280 if protein_method == "mtdp" else 1024,
        condition_dim=4 if condition_method == "rnapsec_condition" else 14,
        embed_dim=256,
        dropout=0.5,
    ).to(device)

    phase_classifier = PhaseClassificationHead(embed_dim=256, dropout=0.5).to(device)

    optimizer = torch.optim.Adam(list(system_encoder.parameters()) + list(phase_classifier.parameters()), lr=7e-4, weight_decay=7e-6)
    epochs = 1000
    best_auroc = 0

    for epoch in range(epochs):
        system_encoder.train()
        phase_classifier.train()
        optimizer.zero_grad()
        # Forward pass on all train data
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        c_train_tensor = torch.tensor(c_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

        x_system = system_encoder(x_train_tensor, c_train_tensor)
        y_probs = phase_classifier(x_system)
        loss = torch.nn.functional.binary_cross_entropy(y_probs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Evaluation on all test data
        system_encoder.eval()
        phase_classifier.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
            c_test_tensor = torch.tensor(c_test, dtype=torch.float32, device=device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

            x_system_test = system_encoder(x_test_tensor, c_test_tensor)
            y_probs_test = phase_classifier(x_system_test).detach().cpu().numpy()
            y_pred_test = (y_probs_test >= 0.5).astype(int)
            test_loss = torch.nn.functional.binary_cross_entropy(torch.tensor(y_probs_test, dtype=torch.float32, device=device), y_test_tensor).item()

        auroc = roc_auc_score(y_test, y_probs_test)
        auprc = average_precision_score(y_test, y_probs_test)
        kappa = cohen_kappa_score(y_test, y_pred_test)
        mcc = matthews_corrcoef(y_test, y_pred_test)

        # Log metrics for every epoch
        swanlab.log({"train_loss": train_loss, "test_loss": test_loss, "auroc": auroc, "auprc": auprc, "kappa": kappa, "mcc": mcc}, step=epoch)

        # Save best model by AUROC
        if epoch == 0 or auroc > best_auroc:
            best_auroc = auroc
            best_state = {
                "protein_encoder": system_encoder.state_dict(),
                "classifier": phase_classifier.state_dict(),
                "epoch": epoch,
                "auroc": auroc,
                "auprc": auprc,
                "kappa": kappa,
                "mcc": mcc,
            }
            torch.save(best_state, f"./task1/models/condformer_{system_name}_{split_method}_best.pth")
            swanlab.log({"best_auroc": best_auroc}, step=epoch)

        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Kappa: {kappa:.4f}, MCC: {mcc:.4f}"
        )

    # Log final params and best metrics
    swanlab.log(
        {
            "system_name": system_name,
            "split_method": split_method,
            "protein_embed_method": protein_method,
            "dna_rna_embed_method": dna_rna_method,
            "condition_embed_method": condition_method,
            "epochs": epochs,
        }
    )


def protein1_regression(
    protein_method="rnapsec_protein", condition_method="rnapsec_condition", transfer_model_path="./task1/models/condformer_protein(1)_random_best.pth"
):
    """Load the DB2 dataset for IDR system and perform regression."""
    from db2 import DB2Dataset
    from meta_config import create_meta_config
    from model1 import Protein1SystemEncoder, PhaseRegressionHead
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    meta_config = create_meta_config(protein=protein_method, condition=condition_method)

    idr_dataset = DB2Dataset(root="./data", name="DB2", embedding_config=meta_config, subset_type="IDR", download=True)

    x = idr_dataset[:].obsm[f"x_{protein_method}"]
    c = idr_dataset[:].obsm[f"c_{condition_method}"]
    y = idr_dataset[:].obsm["y"]

    x = torch.tensor(StandardScaler().fit_transform(x), dtype=torch.float32)
    c = torch.tensor(StandardScaler().fit_transform(c), dtype=torch.float32)

    y = np.array(y)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=42)
    x_train, x_test = x[train_idx], x[test_idx]
    c_train, c_test = c[train_idx], c[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_encoder = Protein1SystemEncoder(
        protein_dim=29 if protein_method == "rnapsec_protein" else 1280 if protein_method == "mtdp" else 1152,
        condition_dim=4 if condition_method == "rnapsec_condition" else 14,
        embed_dim=256,
        dropout=0.5,
    )

    # transfer_model_path = './task1/models/condformer_protein(1)_component_concentration_stratified_best.pth'
    best_state = torch.load(transfer_model_path, weights_only=False)
    system_encoder.load_state_dict(best_state["protein_encoder"])

    # Freeze encoder parameters
    for param in system_encoder.parameters():
        param.requires_grad = False

    # Regression head
    regression_head = PhaseRegressionHead(embed_dim=256, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(regression_head.parameters(), lr=3e-4, weight_decay=5e-6)
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_encoder = system_encoder.to(device)
    regression_head = regression_head.to(device)

    x_train = x_train.to(device)
    c_train = c_train.to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test = x_test.to(device)
    c_test = c_test.to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    best_mse = float("inf")
    for epoch in range(epochs):
        regression_head.train()
        optimizer.zero_grad()
        with torch.no_grad():
            x_emb = system_encoder(x_train, c_train)
        y_pred_train = regression_head(x_emb).squeeze()
        loss = torch.nn.functional.mse_loss(y_pred_train, y_train)
        loss.backward()
        optimizer.step()

        # Evaluation
        regression_head.eval()
        with torch.no_grad():
            x_emb_test = system_encoder(x_test, c_test)
            y_pred_test = regression_head(x_emb_test).squeeze()
            test_loss = torch.nn.functional.mse_loss(y_pred_test, y_test).item()
            mse = test_loss
            r2 = r2_score(y_test.cpu().numpy(), y_pred_test.cpu().numpy())

        if mse < best_mse:
            best_mse = mse
            torch.save(regression_head.state_dict(), "./task1/models/idr_regression_head_best.pth")
            logger.info(f"Saved best model at epoch {epoch + 1} with MSE: {mse:.4f}")

        logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f} MSE: {mse:.4f}, R2: {r2:.4f}")

    # Final prediction on test set
    regression_head.load_state_dict(torch.load("./task1/models/idr_regression_head_best.pth"))
    regression_head.eval()
    with torch.no_grad():
        x_emb_test = system_encoder(x_test, c_test)
        y_pred = regression_head(x_emb_test).detach().squeeze().cpu().numpy()

    # Save predictions and ground truth with model name
    model_name = os.path.basename(transfer_model_path).replace(".pth", "")
    df_pred = pd.DataFrame({"y_true": y_test.cpu().numpy(), "y_pred": y_pred})
    df_pred.to_csv(f"./task1/results/pred_vs_true_{model_name}.csv", index=False)

    final_mse = mean_squared_error(y_test.cpu().numpy(), y_pred)
    final_r2 = r2_score(y_test.cpu().numpy(), y_pred)

    logger.info(f"Regression - MSE on Test: {final_mse:.4f}, R2: {final_r2:.4f}")

    sns.jointplot(x=y_test.detach().cpu().numpy(), y=y_pred, kind="reg", color="blue")
    plt.xlabel("True y (test)")
    plt.ylabel("Predicted y (test)")
    plt.suptitle("Joint Plot: True vs Predicted (Test Set)")
    plt.tight_layout()
    plt.show()


# %% run rnapsec baseline

swanlab.init(project_name="ConditionPhase")

protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_random_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_phase_label_stratified_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_component_concentration_stratified_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_crowding_agent_stratified_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_ionic_strength_stratified_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_pH_stratified_best.pth")
protein1_regression(transfer_model_path="./task1/models/condformer_protein(1)_temperature_stratified_best.pth")
