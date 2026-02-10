import re
from typing import Dict, List, Tuple, Union

import loguru
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from MTDP.mtdp_embedder import MTDP
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, T5Tokenizer

logger = loguru.logger


### helper functions for embedding sequences and pooling tensors, returning tensors/arrays ESM Cambrian embeddings, MTDP embeddings
def tensor_pooling(tensor: torch.Tensor, method: str) -> torch.Tensor:
    """Pool the tensor using the specified method.
    Args:
        tensor (torch.Tensor): The input tensor to be pooled.
        method (str): The pooling method to use ('mean', 'max', 'sum', 'norm', 'prod', 'median').
    Returns:
        torch.Tensor: The pooled tensor.
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    if method == "mean":
        return tensor.mean(dim=1).squeeze(0)
    elif method == "max":
        return tensor.max(dim=1).squeeze(0)
    elif method == "sum":
        return tensor.sum(dim=1).squeeze(0)
    elif method == "norm":
        return torch.nn.functional.normalize(tensor, dim=1).squeeze(0)
    elif method == "prod":
        return tensor.prod(dim=1).squeeze(0)
    elif method == "median":
        return torch.median(tensor, dim=1).squeeze(0)
    else:
        raise ValueError(f"Pooling method '{method}' is not supported. Use one of: ['mean', 'max', 'sum', 'norm', 'prod', 'median']")


def seq2esmc_dict(seqs: List[str], max_length=2048, pooling_method="mean", device="cuda") -> dict:
    """
    Generate a dictionary mapping protein sequences to their ESM Cambrian embeddings.
    Args:
        seqs (List[str]): List of protein sequences to embed.
        max_length (int): Maximum length of the protein sequence for tokenization (default: 2048).
        pooling_method (str): Method to pool the embeddings ('mean', 'max', 'sum', 'norm', 'prod', 'median').
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        dict: Mapping from protein sequence to its embedding tensor.
    """
    logger.info(f"Getting ESM-Cambrian-600M model embeddings for {len(seqs)} sequences ({len(set(seqs))} non-duplicated) and pooling method '{pooling_method}'")

    model_esmc_600m = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_large", trust_remote_code=True).to(device)  # ESM Cambrian model 600M
    # model_esmc_300m = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True) # ESM Cambrian model 300M

    tokenizer = model_esmc_600m.tokenizer
    seq2tensor = {}
    for seq in tqdm(set(seqs)):
        tokenized = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(device)
        with torch.no_grad():
            output = model_esmc_600m(**tokenized)
            protein_tensor = output.last_hidden_state
            pooled_tensor = tensor_pooling(protein_tensor, pooling_method)
            seq2tensor[seq] = pooled_tensor.detach().cpu().numpy()
    return seq2tensor


def seq2mtdp_dict(seqs: List[str], max_length=2048, device="cuda") -> dict:
    """
    Convert protein sequences to MTDP embeddings using a pre-trained MTDP model.

    This function takes a list of protein sequences and generates embeddings using the MTDP
    (Molecular Transformer for Drug Prediction) model. Each sequence is tokenized, processed
    through the model, and converted to a numerical representation.

    Args:
        seqs (List[str]): List of protein sequences to embed
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 2048.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        dict: Dictionary mapping each unique sequence to its MTDP embedding tensor.
            Each embedding has shape (2, 1280) representing 2 sequences with
            1280-dimensional features.

    Note:
        - The function loads a pre-trained MTDP model from './MTDP/models/UniProtKB/UniProtKB.bin'
        - Uses T5Tokenizer from './MTDP/MTDP_tokenizer/'
        - Only processes unique sequences to avoid redundant computation
        - Returns embeddings as detached CPU numpy arrays
    """
    # Load the tokenizer

    tokenizer = T5Tokenizer.from_pretrained("./MTDP/MTDP_tokenizer/", do_lower_case=False)  # ./scripts

    model = MTDP()
    model.load_state_dict(torch.load("./MTDP/models/UniProtKB/UniProtKB.bin")) #./scripts
    model.to(device)

    seq2tensor = {}
    for seq in tqdm(set(seqs)):
        tokenized = tokenizer([seq], add_special_tokens=True, padding="max_length", truncation=True, max_length=max_length)
        data = Dataset.from_dict(tokenized)
        with torch.no_grad():
            input_ids = torch.Tensor(data["input_ids"]).long()
            mask = torch.Tensor(data["attention_mask"]).long()
        embedding_repr = model(input_ids=input_ids.to(device), attention_mask=mask.to(device))

        final_embed = embedding_repr["logits"].detach().cpu().numpy()  # shape: (2, 1280) 2 sequences, each has a 1280-dimensional feature

        seq2tensor[seq] = final_embed.squeeze(0)  # remove the batch dimension, final shape: (1280,)

    return seq2tensor


# helper function adapted from RNAPSEC/predict_condition_new_sequences/prepro_input/prepro_ex_protein_feature.py
# https://github1s.com/ycu-iil/RNAPSEC/blob/HEAD/predict_behavior_new_sequences/prepro_input


def seqs2features29(seqs: List[str]) -> pd.DataFrame:
    """
    Generate a 29-dimensional feature vector for each protein sequence.
    Given a list of protein sequences, computes molecular weight, amino acid composition, aromaticity, instability index, GRAVY,
    average flexibility, isoelectric point, and secondary structure fractions. Returns a DataFrame with one row per sequence and 29 feature columns.
    Adapted from RNAPSEC, check https://github1s.com/ycu-iil/RNAPSEC/blob/HEAD/predict_behavior_new_sequences/prepro_input
    Args:
        seqs (List[str]): List of protein sequences.
    Returns:
        pd.DataFrame: DataFrame of shape (len(seqs), 29) with extracted features.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    # Initialize lists for features
    weights = []
    aa_counts_list = []
    aromaticities = []
    instabilities = []
    gravies = []
    flexibilities = []
    isoelectric_points = []
    sec_structures_list = []

    for seq in seqs:
        # Check if sequence is empty or contains only 'X' characters
        clean_seq = seq.replace("X", "")

        if not clean_seq or len(clean_seq) == 0:
            # Use zeros as placeholders for empty/X-only sequences
            weights.append(0.0)
            aa_counts_list.append({aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"})
            aromaticities.append(0.0)
            instabilities.append(0.0)
            gravies.append(0.0)
            flexibilities.append(0.0)
            isoelectric_points.append(0.0)
            sec_structures_list.append([0.0, 0.0, 0.0])  # helix, turn, sheet
        else:
            # Compute features for valid sequences
            analysis = ProteinAnalysis(clean_seq)
            weights.append(analysis.molecular_weight())
            aa_counts_list.append(analysis.get_amino_acids_percent())
            aromaticities.append(analysis.aromaticity())
            instabilities.append(analysis.instability_index())
            gravies.append(analysis.gravy())
            flexibilities.append(np.mean(analysis.flexibility()))
            isoelectric_points.append(analysis.isoelectric_point())
            sec_structures_list.append(analysis.secondary_structure_fraction())

    # Convert to DataFrames
    aa_counts = pd.DataFrame(aa_counts_list).fillna(0.0).rename(columns=lambda x: f"protein_{x}")
    sec_structures = pd.DataFrame(sec_structures_list, columns=["protein_helix", "protein_turn", "protein_sheet"])

    # Combine all features into a single DataFrame
    protein_features_29 = pd.concat(
        [
            pd.DataFrame(weights, columns=["protein_weight"]),
            aa_counts,
            pd.DataFrame(aromaticities, columns=["protein_aromaticity"]),
            pd.DataFrame(instabilities, columns=["protein_instability"]),
            pd.DataFrame(gravies, columns=["protein_gravy"]),
            pd.DataFrame(flexibilities, columns=["protein_flexibility_average"]),
            pd.DataFrame(isoelectric_points, columns=["protein_isoelectric_point"]),
            sec_structures,
        ],
        axis=1,
    )

    return protein_features_29


def seqs2features97(seqs: List[str]) -> pd.DataFrame:
    """Generate a 97-dimensional feature vector for each RNA sequence.
    Given a list of RNA sequences, computes nucleotide composition, GC content, and other features
    using MathFeature toolkit.

    Args:
        seqs (List[str]): List of RNA sequences.
    Returns:
        pd.DataFrame: DataFrame of shape (len(seqs), 97) with extracted RNA features.
    """
    import os
    import subprocess
    import tempfile

    logger.info(f"Computing 97-dimensional RNA features for {len(seqs)} sequences using MathFeature")

    # Create temporary directory for all operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir = tempfile.mkdtemp()
        # Create input FASTA file
        fasta_file = os.path.join(tmp_dir, "rna_seq.fasta")
        with open(fasta_file, "w") as f:
            for i, seq in enumerate(seqs):
                f.write(f">seq_{i}\n{seq}\n")

        # Create results directory
        result_dir = os.path.join(tmp_dir, "mf_result")
        os.makedirs(result_dir, exist_ok=True)

        # Create input file for k-mer method
        input_file = os.path.join(tmp_dir, "mf_input.txt")
        with open(input_file, "w") as f:
            f.write("3\n")  # 3-mer

        try:
            # Run MathFeature commands
            commands = [
                f"python ./MathFeature/methods/ExtractionTechniques.py -i {fasta_file} -o {result_dir}/mf_3mer.csv -t kmer -seq 2 < {input_file}",
                f"python ./MathFeature/methods/FickettScore.py -i {fasta_file} -o {result_dir}/mf_fickett.csv -seq 2",
                f"python ./MathFeature/methods/CodingClass.py -i {fasta_file} -o {result_dir}/mf_orf.csv",
                f"python ./MathFeature/methods/TsallisEntropy.py -i {fasta_file} -o {result_dir}/mf_tsallis_entropy.csv -k 2 -q 0.1",
                f"python ./MathFeature/methods/EntropyClass.py -i {fasta_file} -o {result_dir}/mf_shannon.csv -k 2 -e Shannon",
                f"python ./MathFeature/methods/FourierClass.py -i {fasta_file} -o {result_dir}/mf_fourier_binary.csv -r 1",
                f"python ./MathFeature/methods/FourierClass.py -i {fasta_file} -o {result_dir}/mf_fourier_zcurve.csv -r 2",
                f"python ./MathFeature/methods/FourierClass.py -i {fasta_file} -o {result_dir}/mf_fourier_real.csv -r 3",
            ]

            for cmd in commands:
                logger.debug(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Command failed: {cmd}")
                    logger.warning(f"Error: {result.stderr}")

            # Process results similar to the provided code

            # Fickett score
            df_fickett_score0 = pd.read_csv(os.path.join(result_dir, "mf_fickett.csv"))
            df_fickett2 = df_fickett_score0.loc[:, ["fickett_score-ORF", "fickett_score-full-sequence"]]
            df_fickett2 = df_fickett2[pd.to_numeric(df_fickett2["fickett_score-ORF"], errors="coerce").notna()]

            # K-mer features
            df_3mer_0 = pd.read_csv(os.path.join(result_dir, "mf_3mer.csv"), index_col=False)
            df_kmer = df_3mer_0[df_3mer_0.AA.notna()]
            df_kmer = df_kmer[pd.to_numeric(df_kmer.A, errors="coerce").notna()].drop(["nameseq", "label"], axis="columns")

            # ORF features
            df_orf = pd.read_csv(os.path.join(result_dir, "mf_orf.csv"))
            df_orf_used = df_orf.cv_ORF_length
            df_orf_used = df_orf_used[pd.to_numeric(df_orf_used, errors="coerce").notna()]

            # Entropy features
            df_tsallis = pd.read_csv(os.path.join(result_dir, "mf_tsallis_entropy.csv"))
            df_tsallis2_1 = df_tsallis.reset_index(drop=True)
            df_tsallis3 = df_tsallis2_1.rename(columns={"k1": "tsallis_k1", "k2": "tsallis_k2"})
            df_tsallis3 = df_tsallis3.loc[:, ["tsallis_k1", "tsallis_k2"]]
            df_tsallis3 = df_tsallis3[pd.to_numeric(df_tsallis3.tsallis_k1, errors="coerce").notna()]

            df_shanon0 = pd.read_csv(os.path.join(result_dir, "mf_shannon.csv"))
            df_shanon = df_shanon0.drop(["nameseq", "label"], axis="columns")
            df_shanon = df_shanon.rename(columns={"k1": "shanon_k1", "k2": "shanon_k2"})
            df_shanon = df_shanon[pd.to_numeric(df_shanon.shanon_k1, errors="coerce").notna()]

            # Fourier features
            df_fourier_binary = pd.read_csv(os.path.join(result_dir, "mf_fourier_binary.csv"))
            df_fourier_real = pd.read_csv(os.path.join(result_dir, "mf_fourier_real.csv"))
            df_fourier_zcurve = pd.read_csv(os.path.join(result_dir, "mf_fourier_zcurve.csv"))

            df_fourier_zcurve = df_fourier_zcurve.rename(columns={"average": "zcurve_avg", "peak": "zcurve_peak"})
            df_fourier_real = df_fourier_real.rename(columns={"average": "real_avg", "peak": "real_peak"})
            df_fourier_binary = df_fourier_binary.rename(columns={"average": "binary_avg", "peak": "binary_peak"})

            df_fourier = pd.concat(
                [
                    df_fourier_real.loc[:, ["real_avg", "real_peak"]],
                    df_fourier_binary.loc[:, ["binary_avg", "binary_peak"]],
                    df_fourier_zcurve.loc[:, ["zcurve_avg", "zcurve_peak"]],
                ],
                axis="columns",
            )
            df_fourier = df_fourier[pd.to_numeric(df_fourier.zcurve_avg, errors="coerce").notna()]

            # Rename k-mer columns
            for col in df_kmer.columns:
                df_kmer = df_kmer.rename(columns={col: f"rna_{col}"})

            # Concatenate all features
            df_rna_features = pd.concat([df_kmer, df_fickett2, df_tsallis3, df_fourier, df_shanon, df_orf_used.to_frame("rna_cv_ORF_length")], axis="columns")

            # Rename all columns to have rna_ prefix
            df_rna_features_final = df_rna_features.copy()
            for col in df_rna_features.columns:
                if not col.startswith("rna_"):
                    df_rna_features_final = df_rna_features_final.rename(columns={col: f"rna_{col}"})

            logger.info(f"Generated RNA features with shape: {df_rna_features_final.shape}")
            return df_rna_features_final

        except Exception as e:
            logger.error(f"Error computing RNA features: {e}")
            # Fallback: return dummy features


def seqs2ind(seqs: List[str], max_len=2048) -> pd.DataFrame:
    """
    Convert sequences of amino acids to indices based on a predefined mapping.
    """
    SEQ2IND = {
        "I": 0,
        "L": 1,
        "V": 2,
        "F": 3,
        "M": 4,
        "C": 5,
        "A": 6,
        "G": 7,
        "P": 8,
        "T": 9,
        "S": 10,
        "Y": 11,
        "W": 12,
        "Q": 13,
        "N": 14,
        "H": 15,
        "E": 16,
        "D": 17,
        "K": 18,
        "R": 19,
        "X": 20,
        "J": 21,
        "*": 22,
        "-": 23,
    }  # J = padding, * = any amino
    # IND2SEQ = {ind: AA for AA, ind in SEQ2IND.items()}

    reps = []
    for seq in tqdm(seqs):
        if len(seq) <= max_len:
            seq += "-" * (max_len - len(seq))
            reps.append([SEQ2IND[i] for i in seq])
        else:
            start = np.random.randint(0, len(seq) - max_len)
            seq = seq[start : start + max_len]
            reps.append([SEQ2IND[i] for i in seq])

    seq2ind_df = pd.DataFrame(reps, columns=[f"seq2ind_{i}" for i in range(max_len)])

    return seq2ind_df


def seq2onehot(seqs: List[str], max_len: int = 2048) -> List[np.ndarray]:
    """
    Convert sequences of amino acids to one-hot encoded arrays of length max_len (pad or truncate).
    All characters other than the 20 standard amino acids are encoded as 'X' (index 20).
    Args:
        seqs (List[str]): List of protein sequences.
        max_len (int): Length to pad or truncate sequences to (default: 2048).
    Returns:
        List[np.ndarray]: List of one-hot encoded arrays, each of shape (max_len, 21).
    """
    AA = "ACDEFGHIKLMNPQRSTVWYX"
    AA2IDX = {aa: idx for idx, aa in enumerate(AA)}
    n_aa = len(AA)

    seqs = [re.sub(f"[^{AA}]", "X", seq.upper()) for seq in seqs]
    onehot_list = []
    for seq in tqdm(seqs):
        seq = seq[:max_len].ljust(max_len, "X")
        indices = [AA2IDX.get(aa, AA2IDX["X"]) for aa in seq]
        onehot = np.eye(n_aa, dtype=np.float32)[indices]
        onehot_list.append(onehot)
    onehot_array = np.array(onehot_list)
    return onehot_array


###################################################################################################
### embedding method for protein component, DNA/RNA component, and experimental condition component
###################################################################################################


def get_protein_embeddings(seqs: List[str], embedding_model="esmc", device="cuda") -> pd.DataFrame:
    """
    Generate ESM Cambrian embeddings for a list of protein sequences.

    Args:
        seqs (List[str]): Protein sequences to embed.
        embedding_model (str): Model identifier (default: 'esmc').

    Returns:
        Dict[str, torch.Tensor]: Mapping from sequence to embedding tensor.
    """

    if embedding_model == "esmc":
        seq2tensor = seq2esmc_dict(seqs, pooling_method="mean", device=device)  # pooling method varies for ESM Cambrian embeddings
        # esmc_tensor = torch.stack([seq2tensor[seq] for seq in seqs], dim=0)
        esmc_df = pd.DataFrame([seq2tensor[seq] for seq in seqs])
        return esmc_df

    elif embedding_model == "mtdp":
        seq2tensor = seq2mtdp_dict(seqs, device=device)
        # mtdp_tensor = torch.stack([torch.tensor(seq2tensor[seq]) for seq in seqs])
        mtdp_df = pd.DataFrame([seq2tensor[seq] for seq in seqs])  # 1280 is the feature dimension of MTDP embeddings
        return mtdp_df

    elif embedding_model == "rnapsec_protein":
        return seqs2features29(seqs)  # Use ESM Cambrian dictionary embeddings as a fallback

    elif embedding_model == "seq2ind":
        seq2ind_df = seqs2ind(seqs)
        return seq2ind_df
    elif embedding_model == "seq2onehot":
        onehot_list = seq2onehot(seqs)
        return onehot_list

    else:
        raise ValueError(f"Embedding model '{embedding_model}' is not supported. Use one of: ['esmc', 'mtdp', 'rnapsec_protein', 'seq2ind', 'seq2onehot']")


def get_dna_rna_embeddings(nucleic_acids: pd.Series, embedding_model="dictionary") -> pd.DataFrame:
    """
    Generate indices for embeddings and index mapping for nucleic acids (DNA/RNA).

    Args:
        nucleic_acids (pd.Series): Series containing nucleic acid descriptions or sequences (DNA/RNA).

    Returns:
        Tuple[torch.Tensor, Dict[str, int]]: A tuple containing:
        - torch.Tensor: Index tensor for DNA/RNA sequences
        - Dict[str, int]: Mapping dictionary from sequence strings to their indices
    """
    if embedding_model == "dictionary":
        # Create unique sequences and their corresponding indices
        logger.info(f"Getting dictionary embeddings for {len(nucleic_acids)} nucleic acids ({len(set(nucleic_acids))} non-duplicated)")
        nucleic2index_mapping = {x: idx for idx, x in enumerate(nucleic_acids.drop_duplicates())}

        # Generate index tensor for all nucleic acids
        nucleic_acids_index_embeddings = pd.Series([nucleic2index_mapping[x] for x in nucleic_acids])  # torch.tensor(, dtype=torch.long).unsqueeze(1)

        return nucleic_acids_index_embeddings, nucleic2index_mapping

    if embedding_model == "rnapsec_rna":
        nucleic_acids_embeddings = seqs2features97(nucleic_acids.tolist())  # Use MathFeature toolkit to get 97-dimensional features
        nucleic_acids_mapping = None

        return nucleic_acids_embeddings, nucleic_acids_mapping


def get_condition_embeddings(conditions: pd.DataFrame, embedding_model="droppler") -> pd.DataFrame:
    """
    Generate embeddings for experimental conditions.
    Args:
        conditions (pd.DataFrame): DataFrame containing experimental conditions.
        embedding_model (str): Model identifier for condition embeddings (default: 'droppler').
    Returns:
        torch.Tensor: Tensor of condition embeddings.
    """

    def temperature_validate(temperature: Union[str, int, float], in_type="one", out_type="one") -> List[float]:
        """
        Validates and parses temperature input into a list of floats or low/high tuples.
            temperature (Union[str, int, float]): Temperature value(s). Can be:
                - str: Single value ("25"), multiple values ("25, 40"), or ranges ("25-40, 50-60").
                - int or float: Single numeric value.
            in_type (str, optional): Input type indicator (default: 'one'). Not used in current implementation.
            out_type (str, optional): Output format.
                - 'scalar': Returns a list of float values.
                - 'low, high': Returns two lists: lows and highs for each range.
            List[float] or Tuple[List[float], List[float]]:
                - If out_type is 'scalar', returns a list of float values.
                - If out_type is 'low, high', returns two lists: lows and highs.
        Raises:
            TypeError: If the input temperature type is unsupported.
        """
        if in_type == "one" and out_type == "one":
            return float(temperature)
        elif in_type == "one" and out_type == "two":
            return float(temperature), float(temperature)
        elif in_type == "two" and out_type == "one":
            return (float(temperature.split(", ")[0]) + float(temperature.split(", ")[1])) / 2
        elif in_type == "two" and out_type == "two":
            return temperature.split(", ")[0], temperature.split(", ")[1]
        else:
            raise TypeError(f"Unsupported temperature type: in_type={in_type}, out_type={out_type}")

    def temperature_to_bins(celsius_low: str, celsius_high: str) -> List[float]:
        """
        Convert temperature to 10-bin representation from 0 to 100°C

        Temperature ranges are represented as probability distributions across 10 bins:
        - Bin 0: 0-10°C
        - Bin 1: 10-20°C
        - ...
        - Bin 9: 90-100°C

        Examples:
        - <40°C becomes [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        - 0-20°C becomes [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        - 25°C becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        - -2°C becomes [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        celsius_low = float(celsius_low)
        celsius_high = float(celsius_high)

        low_ind = int(celsius_low // 10)
        high_ind = int(celsius_high // 10)

        if not (0 <= low_ind <= high_ind <= 9):
            logger.warning(f"Detected out-of-bounds temperature range: {celsius_low} to {celsius_high}. Defaulting to 0-100°C bins.")

        bins = np.zeros(10, dtype=int)
        bins[low_ind : high_ind + 1] = 1

        return bins

    if embedding_model == "droppler":
        logger.info(f"Getting Droppler embeddings for 14 condition dims across {len(conditions)} entries")

        # Extract protein concentration tensor using log10 transformation
        component_concentration = (
            conditions["Protein (DNA/RNA) concentration"].str.replace(" µM", "").str.split(";", expand=True).replace("", "0").fillna("0").astype(float).sum(1)
        )
        component_concentration = (component_concentration + 5e-4).map(np.log10)  # smooth out 0 for log10, epsilon = 1% of non-zero smallest concentration
        # sum across protein and protein fusion constructs like [gfp-protein]
        # component_concentration_tensor = torch.tensor((component_concentration + 5e-4).map(np.log10).values) # smooth out 0 for log10, epsilon = 1% of non-zero smallest concentration
        # component_concentration_tensor.shape shape: (n_entry,)
        # Extract the temperature tensor

        temperature = conditions["Temperature"].apply(lambda x: temperature_to_bins(*temperature_validate(x, in_type="two", out_type="two")))
        temperature = pd.DataFrame(np.vstack(temperature.values), columns=[f"temperature_{i}" for i in range(10)], index=conditions.index)
        # temperature_tensor = torch.tensor(np.vstack(temperature.values))
        # temperature_tensor.shape: (n_entry, 10) temperature_tensor.shape

        # Extract the salt concentration tensor
        ionic_strength = (
            conditions["Ionic strength"].str.replace(" mM", "").str.strip(";").str.split(";", expand=True).fillna(0).astype(float).sum(1)
        )  # sum across salt and salt fusion construct like [gfp-salt]
        ionic_strength = ionic_strength.map(lambda x: 0 if x == 0 else np.log10(x))  # deal with 0 M NaCl
        # ionic_strength_tensor = torch.tensor(ionic_strength.values) # deal with 0 M NaCl
        # ionic_strength_tensor.shape: (n_entry,)

        # Extract the pH tensor
        ph = conditions["Buffer pH"].str.replace("pH ", "").astype(float)
        # ph_tensor = torch.tensor(ph.values)
        # ph_tensor.shape: (n_entry)

        # Extract the crowding agent tensor
        crowding_agent = conditions["Crowding agent"].map({"Yes": 1, "No": 0}).astype(float)
        # crowding_agent_tensor = torch.tensor(crowding_agent.values)
        # crowding_agent_tensor.shape: (n_entry)

        # Combine all tensors into a condition embedding tensor like Droppler style
        # Bioinformatics. 2021 Oct 25;37(20):3473-3479. doi: 10.1093/bioinformatics/btab350.
        # condition_tensor = torch.cat((
        #     temperature_tensor, # shape: (n_entry, 10) [:,0:10]
        #     component_concentration_tensor.unsqueeze(1), # shape: (n_entry, 1) [:,10]
        #     ionic_strength_tensor.unsqueeze(1), # shape: (n_entry,1) [:,11]
        #     crowding_agent_tensor.unsqueeze(1), # shape: (n_entry,1) [:,12]
        #     ph_tensor.unsqueeze(1), # shape: (n_entry,1) [:,:13]
        # ), dim=1).to(torch.float32)
        # condition_tensor.shape: (n_entry, 14) dtype: float64

        condition_df = pd.concat(
            [
                temperature,
                pd.DataFrame(
                    {"component_concentration": component_concentration, "ionic_strength": ionic_strength, "crowding_agent": crowding_agent, "ph": ph}
                ),
            ],
            axis=1,
        )

        return condition_df  # condition_tensor

    elif embedding_model.startswith("rnapsec_condition"):
        logger.info(f"Getting rnapsec_condition embeddings for 5+ condition dims across {len(conditions)} entries")

        # Extract the pH class
        # Map pH to bins: 0-7, 7-8, 8-14
        # ph_tensor.shape: (n_entry)

        ph = conditions["Buffer pH"].str.replace("pH ", "").astype(float)
        ph_class = pd.cut(ph, bins=[0, 7, 8, 14], labels=False, include_lowest=True)

        # Extract the temperature class
        # Map temperature to bins: 0-25, 25-30, 30-40 using pandas cut
        # temperature_tensor.shape: (n_entry)
        temperature = conditions["Temperature"]
        # convert temperature range from DB1 to the average and input for rnapsec format
        temperature = temperature.apply(
            lambda x: temperature_validate(x, in_type="two", out_type="one") if ", " in str(x) else temperature_validate(x, in_type="one", out_type="one")
        )
        temperature_class = pd.cut(temperature, bins=[0, 25, 30, 40, 100], labels=False, include_lowest=True)

        # Extract the salt concentration class
        ionic_strength = conditions["Ionic strength"].str.strip().str.replace("mM", "")
        ionic_strength = ionic_strength.astype(str).str.split(";", expand=True).fillna(0).astype(float).sum(axis=1)

        ionic_strength_log = np.where(ionic_strength == 0, 0, np.log10(ionic_strength))
        ionic_strength_q5 = np.percentile(ionic_strength_log, [20, 40, 60, 80])
        ionic_strength_class = np.digitize(ionic_strength_log, ionic_strength_q5, right=True)

        # Extract the component concentration for protein and DNA/RNA
        component_concentration = (
            conditions["Protein (DNA/RNA) concentration"]
            .astype(str)
            .str.replace("µM", "")
            .str.replace("μM", "")
            .str.replace("uM", "")
            .str.replace(" µM", "")
            .str.replace(" μM", "")
            .str.replace(" uM", "")
            .str.split(";", expand=True)
            .astype(float)
            .fillna(0)
        )

        component_concentration = (component_concentration + 5e-4).map(np.log10)
        component_concentration.columns = [f"conc_{i}" for i in range(component_concentration.shape[1])]

        rnapsec_condition_df = pd.concat(
            [pd.DataFrame({"ph": ph, "temperature": temperature, "ionic_strength": ionic_strength_log}), component_concentration], axis=1
        )

        rnapsec_condition_class_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "ph_class": ph_class,
                        "temperature_class": temperature_class,
                        "ionic_strength_class": ionic_strength_class,
                    }
                ),
                component_concentration,
            ],
            axis=1,
        )

        return rnapsec_condition_class_df if embedding_model.endswith("_class") else rnapsec_condition_df

    else:
        raise ValueError(
            f"Condition embedding model '{embedding_model}' is not supported.\
            Use one of: ['droppler', 'rnapsec_condition', 'rnapsec_condition_class']"
        )
