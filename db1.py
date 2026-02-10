import os
import os.path as osp
import re
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
import anndata as ad

# torch tools
from torch.utils.data import Dataset

import sys
sys.path.append('./scripts')  # Add parent directory to path

from meta_config import MetaEmbeddingConfig, create_meta_config
from meta_embed import (get_condition_embeddings, get_dna_rna_embeddings, get_protein_embeddings)
from meta_plot import plot_system_upset


# %% Dataset

class DB1Dataset(Dataset):
    """
    PyTorch Dataset for LLPSDBv2 (DB1). Handles downloading, processing, filtering, and embedding of LLPSDBv2 data for 
    various system types (e.g., protein-only, protein+RNA/DNA).
    Main features:
    - Downloads and extracts LLPSDBv2 data if needed.
    - Processes and validates raw experimental/protein data.
    - Filters by system type and missing values.
    - Embeds proteins, nucleic acids, and conditions using configurable models.
    - Saves processed/embedded data as AnnData (.h5ad).
    - Supports multiple train/test split strategies.
    - Implements standard Dataset interface.

    Args:
        root (str): Root directory for data.
        name (str): Dataset name.
        system (str): System type (e.g., 'protein(1)', 'protein(2) + RNA').
        embedding_config (MetaEmbeddingConfig): Embedding configuration.
        download (bool): Download data if not found.

    Example:
        dataset = DB1Dataset(root='./data', system='protein(1)', embedding_config=my_config)
        x, c, y = dataset.get_tensor()
    """
    # LLPSDB v2 official download base URL

    # Available zip files for download
    download_files = {
        'Phase_separation_ambiguous': 'Phase_separation_ambiguous.zip',
        'No_phase_separation_ambiguous': 'No_phase_separation_ambiguous.zip',
        'Phase_separation_Unambiguous': 'Phase_separation_Unambiguous.zip',
        'No_phase_separation_Unambiguous': 'No_phase_separation_Unambiguous.zip'
    }

    # Expected file structure after extraction
    expected_files = [
        "Phase_separation_Unambiguous/LLPS.xlsx",
        "Phase_separation_Unambiguous/protein.xlsx",

        "Phase_separation_ambiguous/LLPS.xlsx",
        "Phase_separation_ambiguous/protein.xlsx",

        "No_phase_separation_Unambiguous/LLPS.xlsx",
        "No_phase_separation_Unambiguous/protein.xlsx",

        "No_phase_separation_ambiguous/LLPS.xlsx",
        "No_phase_separation_ambiguous/protein.xlsx"
    ]

    def __init__(
        self,
        root: str = '.',
        name: str = 'DB1',
        system: str = None,
        embedding_config: MetaEmbeddingConfig = None,
        download: bool = True,

    ):
        """
        Parsing LLPSDBv2 Dataset for conditionLLPS_DB1 as TorchDataset

        Args:
            root (str): Root directory where the dataset should be saved. Defaults to '.'.
            name (str): The name of the dataset. Defaults to 'LLPSDBv2'.
            system (str): The system type to filter for (e.g., 'protein(1)', 'protein(2) + RNA'). Defaults to None.
            embedding_config (ExperimentEmbeddingConfig): Configuration for protein and condition embeddings. Defaults to None.
            download (bool): Whether to download data if not found. Defaults to True.
        """
        self.root = root
        self.name = name
        self.system = system
        self.embedding_config = embedding_config
        super().__init__()

        # Check if system is valid
        assert self.system in self.system_list, f"System `{self.system}` is not in the list of available systems: {self.system_list}"

        # Download data if needed
        if download and not self._check_raw_data_exists():
            self.download()

        # Check if processed data exists, if not, process raw data
        # Then check if system data exists, if not, process system data

        parsed_exist, system_exist = self._check_processed_data_exists()
        if not parsed_exist:
            self.process_raw()
        if not system_exist:
            self.process_system()

        # Check if embedding data exists, if not, embed the data
        if not self._check_embedding_data_exists():
            self.embed_system()

        logger.info(f"LLPSDB2Dataset initialized with root: {self.root}, name: {self.name}")

    @property
    def system_list(self) -> List[str]:
        """Return the list of systems in the dataset"""
        protein_systems = ['protein(1)', 'protein(2)', 'protein(3)', 'protein(4)', 'protein(5)', 'protein(6)', 'protein(7)']
        protein_rna_systems = ['protein(1) + RNA', 'protein(2) + RNA', 'protein(3) + RNA',]
        protein_dna_systems = ['protein(1) + DNA', 'protein(2) + DNA', 'protein(3) + DNA', 'protein(4) + DNA']
        no_systems = [np.nan]

        return protein_systems + protein_rna_systems + protein_dna_systems + no_systems

    @property
    def raw_dir(self) -> str:
        """Return the raw directory"""
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        """Return the processed directory"""
        return osp.join(self.root, self.name, 'processed')

    @property
    def embedding_dir(self) -> str:
        """Return the embedding directory"""
        return osp.join(self.root, self.name, 'embedding')

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names. Used to check if files exist before downloading."""
        return [osp.join(self.raw_dir, file) for file in self.expected_files]

    @property
    def processed_file_names(self) -> List[str]:
        """Return the processed file names"""
        parsed_raw_files = ["exp_raw.txt", "exp_parsed.txt" ,"exp_protein.txt"] # adjust when more data is processed
        parsed_system_files = [f'{self.system}.no_missing.txt', f'{self.system}.txt'] # adjust when more data is processed

        return parsed_raw_files, parsed_system_files

    @property
    def embedding_file_names(self) -> List[str]:
        """Return the embedding file names"""
        parsed_embeddings_files = [f'{self.system}.no_missing.h5ad']
        return parsed_embeddings_files

    def download(self):
        """Download and extract LLPSDB2 data from LLPSDBv2 website"""
        if self._check_raw_data_exists():
            logger.info(f"{len(self.expected_files)} Files already exist in {self.raw_dir}, skip download.")
            return

        logger.info("Downloading LLPSDB2 data from LLPSDBv2 website...")
        # Create directories
        self._create_directories()

        # Download each required zip file
        for _, zip_filename in self.download_files.items():
            zip_path = osp.join(self.raw_dir, zip_filename) # zip_path = "./LLPSDBv2/raw/Phase_separation_Ambiguous.zip"

            try:
                logger.info(f"Downloading {zip_filename}...")

                # Simulate the download request with proper parameters
                # This may need to be adjusted based on actual website behavior
                import urllib.parse

                # Prepare POST data to trigger download
                post_data = urllib.parse.urlencode({'file': zip_filename}).encode('utf-8')

                # Since the website requires clicking the file name to download,
                # we simulate a GET request to the download URL with the file parameter.
                # This may work if direct download links are allowed.

                # Construct the direct download URL
                direct_url = f"http://bio-comp.org.cn/llpsdbv2/download/{zip_filename}"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'http://bio-comp.org.cn/llpsdbv2/download.php'
                }

                request = urllib.request.Request(direct_url, headers=headers)

                with urllib.request.urlopen(request) as response:
                    with open(zip_path, 'wb') as f:
                        shutil.copyfileobj(response, f)

                logger.info(f"Downloaded {zip_filename} to {zip_path}")

                # Extract zip file
                self._extract_specific_zip(zip_path, self.raw_dir)

                # Rename any .xls files to .xlsx after extraction since the format error in LLPSDB2
                self._rename_xls_to_xlsx()

                # Clean up zip file
                os.unlink(zip_path)

            except Exception as e:
                logger.error(f"Error downloading {zip_filename}: {e}")
                logger.warning(f"Please manually download {zip_filename} from http://bio-comp.org.cn/llpsdbv2/download.php")
                logger.warning(f"and place it in {self.raw_dir}")

                # Check if zip file exists for manual extraction
                if osp.exists(zip_path):
                    try:
                        self._extract_specific_zip(zip_path, self.raw_dir)
                        os.unlink(zip_path)
                    except Exception as extract_error:
                        logger.error(f"Error extracting {zip_filename}: {extract_error}")

        logger.success("Download and extraction completed!")

    def process_raw(self):
        """Load and process the complete dataset without splitting"""

        # Load combined exp and protein data
        logger.info("Processing raw data...")
        df_exp, df_protein = self._load_raw_data()
        df_exp, df_protein = self._validate_fix_raw_data(df_exp, df_protein)

        logger.info("Parsing phase separation labels")
        phase_label = df_exp['Phase separation']# self._extract_phase_labels(df_exp['Phase separation'])

        logger.info("Parsing phase systems by compositions")
        phase_system = df_exp['Components type']
        # logger.debug(f"Protein ID value counts:\n{df_exp['Protein ID'].value_counts()}")

        logger.info("Parsing protein sequences")
        phase_protein_id = df_exp['Protein ID']
        phase_protein_seqs = self._extract_protein_sequences(df_exp, df_protein)

        logger.info("Parsing protein types")
        phase_protein_nd = df_exp['Protein type (N/D)']
        # logger.debug(f"Protein type (N/D) value counts:\n{phase_protein_nd.value_counts()}")

        logger.info("Parsing protein folding states")
        phase_protein_fold = df_exp['Protein structure type']
        # logger.debug(f"Protein structure type value counts:\n{phase_protein_fold.value_counts()}")

        logger.info("Parsing condition temperature")
        phase_temperature = df_exp['Temperature'].apply(self._extract_temperature)

        logger.info("Parsing condition component concentration")
        phase_component_concentration = df_exp['Solute concentration'].apply(self._extract_component_concentration)

        logger.info("Parsing condition ionic strength")
        phase_ionic_strength = df_exp['Salt concentration'].apply(self._extract_ionic_strength)

        logger.info("Parsing condition buffer pH")
        phase_buffer_ph = df_exp['Buffer'].apply(self._extract_buffer_pH)

        logger.info("Parsing condition crowding agent")
        phase_crowding_agent = df_exp['Crowding agent'].apply(self._extract_crowding_agent)

        logger.info("Parsing nucleic acid type")
        phase_nucleic_acid = df_exp['Nucleic acid'].str.strip()

        df_parsed = pd.DataFrame({
            'Phase label': phase_label,
            'Phase system': phase_system,
            'Protein ID': phase_protein_id,
            'Protein type (N/D)': phase_protein_nd,
            'Protein structure type': phase_protein_fold,
            'Protein sequences': phase_protein_seqs,
            'Protein (DNA/RNA) concentration': phase_component_concentration,
            'Nucleic acid': phase_nucleic_acid,
            'Temperature': phase_temperature,
            'Ionic strength': phase_ionic_strength,
            'Buffer pH': phase_buffer_ph,
            'Crowding agent': phase_crowding_agent
        })

        # save processed raw files to the processed directory
        df_exp.to_csv(osp.join(self.processed_dir, 'exp_raw.txt'), sep='\t',)
        df_parsed.to_csv(osp.join(self.processed_dir, 'exp_parsed.txt'), sep='\t')
        df_protein.to_csv(osp.join(self.processed_dir, 'exp_protein.txt'), sep='\t')
        logger.success("Parsed data saved to processed directory.")

    def process_system(self):
        """Load and process the complete dataset by splitting parse data into phase systems"""

        logger.info(f"Processing protein embeddings for system: {self.system} with no missing values (strict)...")
        df_parsed = pd.read_csv(osp.join(self.processed_dir, 'exp_parsed.txt'), sep='\t', index_col=0)

        df_system = self._get_filtered(df_parsed, system_filter=self.system, is_strict=True)
        df_system_loose = self._get_filtered(df_parsed, system_filter=self.system, is_strict=False)
        df_system.to_csv(osp.join(self.processed_dir, f'{df_system.index.name}.txt'), sep='\t',)
        df_system_loose.to_csv(osp.join(self.processed_dir, f'{df_system_loose.index.name}.txt'), sep='\t',)

        # Generate statistics and summary plot
        plot_system_upset(df_system_loose, self.processed_dir)

        logger.success(f"Processing {self.system} completed!")

    def embed_system(self):
        """
        Process protein and condition embeddings for different system types and perform serialization.

        Handles multiple protein systems (protein(1) through protein(7)), 
        protein(?) + RNA systems, and protein(?) + DNA systems. 
        For multi-protein in systems, creates separate embeddings for each protein component.
        For RNA and DNA in systems, processes the nucleic acid names and gives an index tensor for further embedding.

        Saves:
            A .pt file containing a dictionary with:
            - For protein(1): {'x': protein_embeddings, 'c': condition_embeddings, 'y': labels}
            - For protein(2): {'x1': protein1_embeddings, 'x2': protein2_embeddings, 'c': condition_embeddings, 'y': labels}
            - For protein(3): {'x1': protein1_embeddings, 'x2': protein2_embeddings, 'x3': protein3_embeddings, 'c': condition_embeddings, 'y': labels}
            - For protein(1) + RNA: {'x': protein_embeddings, 'x_rna': rna_index, 'c': condition_embeddings, 'y': labels}
            - For protein(1) + DNA: {'x1': protein1_embeddings, 'x_dna': dna_index, 'c': condition_embeddings, 'y': labels}
            - For protein(2) + RNA: {'x1': protein1_embeddings, 'x2': protein2_embeddings, 'x_rna': rna_index, 'c': condition_embeddings, 'y': labels}
            - For protein(2) + DNA: {'x1': protein1_embeddings, 'x2': protein2_embeddings, 'x_dna': dna_index, 'c': condition_embeddings, 'y': labels}

            - For systems with DNA and RNA, the nucleic acid dictionary mapping names to indices is saved as well for reference
        """
        # Parse system to determine components
        df_system = pd.read_csv(osp.join(self.processed_dir, f'{self.system}.no_missing.txt'), sep='\t', index_col=0)
        num_proteins, has_rna, has_dna, has_nucleic_acid = self._extract_systems_components(self.system)

        logger.info(f"Embedding `{df_system.index.name}` --- {num_proteins=}, {has_dna=}, {has_rna=}, {has_nucleic_acid=}")

        # Initialize embeddings dictionary
        embeddings_dict = {}
        protein_embedding_model = self.embedding_config.protein_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding
        condition_embedding_model = self.embedding_config.condition_embedding

        # embedding phase labels
        phase_label = df_system['Phase label'].map({'Yes':1, "No":0})
        # phase_label_tensor = torch.tensor(phase_label.values).unsqueeze(1)  # shape: (n_entry, 1)
        embeddings_dict['y'] = phase_label

        # embedding protein sequences
        if num_proteins == 1:
            seqs = df_system['Protein sequences'].tolist()
            assert all(seqs), "Protein sequences cannot be empty for single protein systems."

            protein_embeddings = get_protein_embeddings(seqs, protein_embedding_model)
            # protein_embeddings = protein_embeddings.detach().cpu() if protein_embeddings.is_cuda else protein_embeddings.detach()
            embeddings_dict[f'x_{protein_embedding_model}'] = protein_embeddings

        if num_proteins > 1:
            for i in range(1, num_proteins+1):
                seqs = df_system['Protein sequences'].str.split(';', expand=True).iloc[:, i-1].tolist()
                # assert all(seqs_for_tensor), f"Protein sequences cannot be empty for multi-protein systems."

                protein_embeddings = get_protein_embeddings(seqs, protein_embedding_model)
                # protein_embeddings = protein_embeddings.detach().cpu() if protein_embeddings.is_cuda else protein_embeddings.detach()
                embeddings_dict[f'x_{i}_{protein_embedding_model}'] = protein_embeddings

        # embedding nucleic acids
        if has_dna:
            dna = df_system['Nucleic acid']
            dna_embeddings, dna_mapping =  get_dna_rna_embeddings(dna, dna_rna_method) # pd.Series, dict
            embeddings_dict[f'x_dna_{dna_rna_method}'] = dna_embeddings
            if dna_mapping is not None:
                embeddings_dict[f'dna_mapping_{dna_rna_method}'] = dna_mapping

        if has_rna:
            rna = df_system['Nucleic acid']
            rna_embeddings, rna_mapping =  get_dna_rna_embeddings(rna, dna_rna_method)
            embeddings_dict[f'x_rna_{dna_rna_method}'] = rna_embeddings
            if rna_mapping is not None:
                embeddings_dict[f'rna_mapping_{dna_rna_method}'] = rna_mapping

        # embedding conditions
        conditions = df_system[['Temperature', 'Protein (DNA/RNA) concentration', 'Ionic strength', 'Buffer pH', 'Crowding agent']]
        condition_embeddings = get_condition_embeddings(conditions, condition_embedding_model)
        # condition_embeddings = condition_embeddings.detach().cpu() if condition_embeddings.is_cuda else condition_embeddings.detach()
        embeddings_dict[f'c_{condition_embedding_model}'] = condition_embeddings
        logger.debug(f"Check Condition tensor NaN: {condition_embeddings.isna().sum().sum()}, Inf: {condition_embeddings.isin([np.inf, -np.inf]).sum().sum()}")


        # save embedding dict and log the structure of saved tensors
        # tensor_file = osp.join(self.processed_dir, f'{df_system.index.name}.{protein_embedding_model}_{condition_embedding_model}.pt')
        # tensor_structure = {k:v.shape if isinstance(v,torch.Tensor) else (type(v),len(v)) for k, v in embeddings_dict.items()}
        # torch.save(embeddings_dict, tensor_file)

        # logger.info(f"Saved embeddings to {tensor_file}")
        # logger.info(f"Tensor structure: {tensor_structure}")

        h5ad_file = osp.join(self.embedding_dir, f'{self.system}.no_missing.h5ad')
        h5ad_file_exists = osp.exists(h5ad_file)

        logger.info(f"Saving embeddings to new {h5ad_file}...") if not h5ad_file_exists else logger.info(f"Modifying embeddings to existing {h5ad_file}...")
        adata = ad.AnnData(X=df_system.astype(str)) if not h5ad_file_exists else ad.read_h5ad(h5ad_file)

        for k, v in embeddings_dict.items():
            if k not in adata.obsm and k not in adata.uns:
                if 'mapping' not in k:
                    adata.obsm[k] = v.values if hasattr(v, 'values') else v
                else:
                    adata.uns[k] = v
            else:
                logger.warning(f"Key {k} already exists in AnnData object, skipping update.")
        adata.write_h5ad(h5ad_file)

        logger.info(f"Saved AnnData structure: {adata} at {h5ad_file}")

    def get_idx_split(self, split_method: str = 'random', seed: Optional[int] = None) -> Tuple[np.array, np.array]:
        """
        Get the split indices for the whole dataset.

        Args:
            split_method (str): Method to split the dataset. Options: 'random', 'stratified', 'kfold'.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            Tuple of train and test indices as tensors.

        """

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        h5ad_file = osp.join(self.embedding_dir, f'{self.system}.no_missing.h5ad')
        adata = ad.read_h5ad(h5ad_file)
        adata_df = adata.to_df()

        # Extract embeddings from AnnData
        y_key = 'y'
        c_key = 'c_{self.embedding_config.condition_embedding}'

        y = adata.obsm[y_key]
        c = adata.obsm[c_key]

        phase_label = adata_df['Phase label']
        temperature = adata_df['Temperature'].str.split(', ', expand=True).astype(float).mean(axis=1)
        component_concentration = adata_df['Protein (DNA/RNA) concentration'].str.split(' ', expand=True)[0].astype(float).fillna(0)
        ionic_strength = adata_df['Ionic strength'].str.split(' ', expand=True)[0].astype(float).fillna(0)
        crowding_agent = adata_df['Crowding agent']
        pH = adata_df['Buffer pH'].str.split(' ', expand=True)[1].astype(float).fillna(0)

        # split method for ablation study in training
        match split_method:
            case 'random':
                idx = np.random.permutation(len(y))
                train_idx = idx[:int(0.8 * len(idx))]
                test_idx = idx[int(0.8 * len(idx)):]

            case 'phase_label_stratified':
                train_idx, test_idx = train_test_split(range(len(y)), stratify=phase_label, test_size=0.2)

            case 'temperature_stratified':
            # Parse temperature string representation and convert to list, then find argmax (see the defnition of _extract_temperature)
                temperature_bins = pd.cut(temperature, [-25, 25, 50, 75, 100], labels=['-25-25°C', '25-50°C', '50-75°C', '75-100°C'])
                train_idx, test_idx = train_test_split(range(len(y)), stratify=temperature_bins, test_size=0.2)
                logger.info(f"Stratified split by temperature bins: {temperature_bins.value_counts()}")

            case 'component_concentration_stratified':
            # Binize component concentration before stratification by 5 quantiles
                c_component_concentration_qbins, qbins = pd.qcut(component_concentration, 5, labels=['low', 'low-medium', 'medium', 'medium-high', 'high'], retbins=True)
                train_idx, test_idx = train_test_split(range(len(y)), stratify=c_component_concentration_qbins, test_size=0.2)
                logger.info(f"Stratified split by component concentration bins: {c_component_concentration_qbins.value_counts()}")
                logger.info(f"Component concentration 5 quantile bins: {qbins}")

            case 'ionic_strength_stratified':
            # Binize ionic strength before stratification by 5 quantiles
                ionic_strength_qbins, qbins = pd.qcut(ionic_strength, 5, labels=['low', 'low-medium', 'medium', 'medium-high', 'high'], retbins=True)
                train_idx, test_idx = train_test_split(range(len(y)), stratify=ionic_strength_qbins, test_size=0.2)
                logger.info(f"Stratified split by ionic strength bins: {ionic_strength_qbins.value_counts()}")
                logger.info(f"log10 Ionic strength 5 quantile bins: {qbins}")

            case 'crowding_agent_stratified':
            # Map crowding agent to binary values and stratify
                train_idx, test_idx = train_test_split(range(len(y)), stratify=crowding_agent, test_size=0.2)
                logger.info(f"Stratified split by crowding agent: {pd.Series(crowding_agent).value_counts()}")

            case 'pH_stratified':
                ph_bins = pd.cut(pH, bins=[2.5,5.5,8.5,11.5], labels=[f'({i},{i+3}]' for i in [2.5,5.5,8.5]])
                train_idx, test_idx = train_test_split(range(len(y)), stratify=ph_bins, test_size=0.2)
                logger.info(f"Stratified split by pH bins: {ph_bins.value_counts()}")

            case _:
                raise ValueError(f"Unknown split method: {split_method}. Available methods: 'random', 'phase_label_stratified', 'temperature_stratified', 'component_concentration_stratified', 'ionic_strength_stratified', 'crowding_agent_stratified', 'pH_stratified'.")

        # train_idx = torch.tensor(train_idx)
        # test_idx = torch.tensor(test_idx)
        return train_idx, test_idx

    def get_tensor(self):
        """
        Get item by index from h5ad anndata file
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            Tuple containing protein embeddings, condition embeddings, and phase labels as tensors
        """
        h5ad_file = osp.join(self.embedding_dir, f'{self.system}.no_missing.h5ad')
        adata = ad.read_h5ad(h5ad_file)

        # Extract embeddings from AnnData obsm
        protein_embedding_model = self.embedding_config.protein_embedding
        condition_embedding_model = self.embedding_config.condition_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding

        y_key = 'y'
        c_key = 'c_{condition_embedding_model}'

        y = torch.tensor(adata.obsm[y_key], dtype=torch.float32)
        c = torch.tensor(adata.obsm[c_key], dtype=torch.float32)

        match self.system:
            case 'protein(1)':
                x_key = f'x_{protein_embedding_model}'
                x = torch.tensor(adata.obsm[x_key], dtype=torch.float32)
                return x, c, y
            case 'protein(2)':
                x1_key = f'x_1_{protein_embedding_model}'
                x2_key = f'x_2_{protein_embedding_model}'
                x1 = torch.tensor(adata.obsm[x1_key], dtype=torch.float32)
                x2 = torch.tensor(adata.obsm[x2_key], dtype=torch.float32)
                return x1, x2, c, y
            case 'protein(3)':
                x1_key = f'x_1_{protein_embedding_model}'
                x2_key = f'x_2_{protein_embedding_model}'
                x3_key = f'x_3_{protein_embedding_model}'
                x1 = torch.tensor(adata.obsm[x1_key], dtype=torch.float32)
                x2 = torch.tensor(adata.obsm[x2_key], dtype=torch.float32)
                x3 = torch.tensor(adata.obsm[x3_key], dtype=torch.float32)
                return x1, x2, x3, c, y
            case 'protein(1) + DNA':
                x_key = f'x_{protein_embedding_model}'
                x_dna_key = f'x_dna_{dna_rna_method}'
                x = torch.tensor(adata.obsm[x_key], dtype=torch.float32)
                x_dna = torch.tensor(adata.obsm[x_dna_key], dtype=torch.float32)
                return x, x_dna, c, y
            case 'protein(1) + RNA':
                x_key = f'x_{protein_embedding_model}'
                x_rna_key = f'x_rna_{dna_rna_method}'
                x = torch.tensor(adata.obsm[x_key], dtype=torch.float32)
                x_rna = torch.tensor(adata.obsm[x_rna_key], dtype=torch.float32)
                return x, x_rna, c, y
            case 'protein(2) + DNA':
                x1_key = f'x_1_{protein_embedding_model}'
                x2_key = f'x_2_{protein_embedding_model}'
                x_dna_key = f'x_dna_{dna_rna_method}'
                x1 = torch.tensor(adata.obsm[x1_key], dtype=torch.float32)
                x2 = torch.tensor(adata.obsm[x2_key], dtype=torch.float32)
                x_dna = torch.tensor(adata.obsm[x_dna_key], dtype=torch.float32)
                return x1, x2, x_dna, c, y
            case 'protein(2) + RNA':
                x1_key = f'x_1_{protein_embedding_model}'
                x2_key = f'x_2_{protein_embedding_model}'
                x_rna_key = f'x_rna_{dna_rna_method}'
                x1 = torch.tensor(adata.obsm[x1_key], dtype=torch.float32)
                x2 = torch.tensor(adata.obsm[x2_key], dtype=torch.float32)
                x_rna = torch.tensor(adata.obsm[x_rna_key], dtype=torch.float32)
                return x1, x2, x_rna, c, y


    ### --------------------------------------- ###
    ### Built-in helper methods for the dataset ###
    ### --------------------------------------- ###

    def _check_raw_data_exists(self) -> bool:
        """Check if required data files exist"""
        raw_exist = all(osp.exists(file) for file in self.raw_file_names)
        logger.info(f"Checking raw files in {self.raw_dir}...= {raw_exist}\n\
                    all of {self.raw_file_names}")

        return all(osp.exists(file) for file in self.raw_file_names)

    def _check_processed_data_exists(self) -> bool:
        """Check if processed data files exist"""

        parsed_raw_files, parsed_system_files = self.processed_file_names
        parsed_exist = all(osp.exists(osp.join(self.processed_dir, file)) for file in parsed_raw_files)
        system_exist = all(osp.exists(osp.join(self.processed_dir, file)) for file in parsed_system_files)

        logger.info(f"Checking processed raw files in {self.processed_dir}...={parsed_exist}\n\
                    all of {parsed_raw_files}")
        logger.info(f"Checking processed system files in {self.processed_dir}...={system_exist}\n\
                    all of {parsed_system_files}")

        return parsed_exist, system_exist

    def _check_embedding_data_exists(self) -> bool:
        """Check if embedding data files exist"""
        protein_method = self.embedding_config.protein_embedding
        condition_method = self.embedding_config.condition_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding

        h5ad_file = osp.join(self.embedding_dir, f'{self.system}.no_missing.h5ad')
        embedding_file_exist = osp.exists(h5ad_file)
        embedding_data_exist = False

        if embedding_file_exist:
            adata = ad.read_h5ad(h5ad_file)
            obsm_keys = adata.obsm.keys()

            *_, has_nucleic_acid = self._extract_systems_components(self.system)

            protein_key_exist = any(protein_method in k for k in obsm_keys if k.startswith('x_'))
            condition_key_exist = any(condition_method in k for k in obsm_keys if k.startswith('c_')) 
            dna_rna_key_exist = any(dna_rna_method in k for k in obsm_keys if k.startswith('x_dna_') or k.startswith('x_rna_')) if has_nucleic_acid else True
            embedding_data_exist = protein_key_exist and condition_key_exist and dna_rna_key_exist

            logger.debug("Checking embedding data keys in h5ad file...")
            logger.debug(f"Protein embedding key exists: {protein_key_exist} ({protein_method})")
            logger.debug(f"Condition embedding key exists: {condition_key_exist} ({condition_method})")
            logger.debug(f"DNA/RNA embedding key exists: {dna_rna_key_exist} ({dna_rna_method}) Default as True when no nucleic acid in the system:")
            logger.debug(f"system has nucleic acid: {has_nucleic_acid}")

        return embedding_file_exist and embedding_data_exist

    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from raw directory, return all_experiments and all_proteins DataFrames"""
        try:
            # Load all exps from the expected files:
            phase_uambi_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[0])) # 'Phase_separation_Unambiguous/LLPS.xlsx'
            phase_ambi_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[2])) # 'Phase_separation_ambiguous/LLPS.xlsx'
            nophase_uambi_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[4])) # 'No_phase_separation_Unambiguous/LLPS.xlsx'
            nophase_ambi_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[6])) # 'No_phase_separation_ambiguous/LLPS.xlsx'
            all_phase_df = pd.concat([phase_uambi_df, phase_ambi_df, nophase_uambi_df, nophase_ambi_df], ignore_index=True)

            # Load protein data from the expected files:
            phase_unambi_protein_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[1])) # 'Phase_separation_Unambiguous/protein.xlsx'
            phase_ambi_protein_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[3])) # 'Phase_separation_ambiguous/protein.xlsx'   
            nophase_unambi_protein_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[5])) # 'No_phase_separation_Unambiguous/protein.xlsx'
            nophase_ambi_protein_df = pd.read_excel(osp.join(self.raw_dir, self.expected_files[7])) # 'No_phase_separation_ambiguous/protein.xlsx'
            all_protein_df = pd.concat([phase_unambi_protein_df, phase_ambi_protein_df, nophase_unambi_protein_df, nophase_ambi_protein_df], ignore_index=True)
            all_protein_df.drop_duplicates(inplace=True)

            return all_phase_df, all_protein_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _get_filtered(self, df_parsed, system_filter="protein(1)", is_strict=True):
        """Get indices by system and missing value filters"""

        # number of proteins in the system
        num_protein, has_rna, has_dna, _ = self._extract_systems_components(system_filter)
        num_component = num_protein + int(has_dna) + int(has_rna)

        # Check if the system is valid
        phase_label_check = df_parsed['Phase label'].isin(['Yes', 'No'])
        phase_system_check = df_parsed['Phase system'] == system_filter
        protein_id_check = df_parsed['Protein ID'].str.split(';').apply(lambda x: len(x) if isinstance(x,list) else 0) <= num_protein
        protein_type_check = df_parsed['Protein type (N/D)'].str.split(';').apply(lambda x: len(x) if isinstance(x,list) else 0) <= num_protein
        protein_sequence_check = df_parsed['Protein sequences'].str.split(';').apply(lambda x: len(x) if isinstance(x,list) else 0) == num_protein
        component_concentration_check = df_parsed['Protein (DNA/RNA) concentration'].str.split(';').apply(lambda x: len(x) if isinstance(x,list) else 0) <= num_component
        nucleic_acid_check = ~df_parsed['Nucleic acid'].isin(['', ' ', '-',np.nan, None]) if 'DNA' in system_filter or 'RNA' in system_filter else\
                        pd.Series([True] * len(df_parsed), index=df_parsed.index)
        temperature_check = df_parsed['Temperature'].notna()
        ionic_strength_check = ~df_parsed['Ionic strength'].isin(['', np.nan, None]) # df_parsed['Ionic strength'].value_counts()
        buffer_ph_check = ~df_parsed['Buffer pH'].isin(['', np.nan, None]) # df_parsed['Buffer pH'].value_counts()
        crowding_agent_check = ~df_parsed['Crowding agent'].isin(['', np.nan, None]) # df_parsed['Crowding agent'].value_counts()

        # Combine all checks
        if is_strict:
            all_check = (phase_label_check & phase_system_check & protein_id_check &
                            protein_type_check & protein_sequence_check & component_concentration_check &
                            nucleic_acid_check & temperature_check & ionic_strength_check &
                            buffer_ph_check & crowding_agent_check)
        else:
            all_check = phase_system_check

        # Filter DataFrame based on all checks
        df_system = df_parsed[all_check]
        df_system.index.name = system_filter + ".no_missing" if is_strict else system_filter

        return df_system

    def _rename_xls_to_xlsx(self):
        """Rename .xls files to .xlsx and remove old .xls files after extraction."""
        for expected_file in self.expected_files:
            xls_path = osp.join(self.raw_dir, expected_file[:-1])  # .xlsx -> .xls
            xlsx_path = osp.join(self.raw_dir, expected_file)
            if osp.exists(xls_path):
                os.rename(xls_path, xlsx_path)
                logger.info(f"Renamed {xls_path} to {xlsx_path}")

    def _extract_specific_zip(self, zip_path: str, target_dir: str):
        """Extract specific zip file for"""
        import zipfile

        logger.info(f"Extracting {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files to the raw directory
            zip_ref.extractall(target_dir)

    def _extract_protein_sequences(self, df_exp: pd.DataFrame, df_protein: pd.DataFrame) -> pd.Series:
        """Extract protein sequences from the DataFrame"""
        # Merge experiment and protein data on 'Protein name'
        protein_id2seq = df_protein.set_index('PID')['Sequence'].str.split(';').apply(lambda x: ''.join(x[1:])).to_dict()

        def map_seq(pid):
            """Map protein ID to sequence using pre-built dict"""
            if pd.isna(pid):
                # Handle missing protein sequence
                return None

            if isinstance(pid, str) and ';' not in pid:
                # Single protein ID
                return protein_id2seq.get(pid, None)

            if isinstance(pid, str) and ';' in pid:
                # Handle multiple protein IDs
                return ';'.join([protein_id2seq.get(x, '') for x in pid.split(';')])

        return df_exp['Protein ID'].apply(map_seq)

    def _extract_phase_labels(self, phase_col: pd.Series) -> pd.Series:
        """Extract binary phase labels from the 'Phase separation' column"""
        # Map 'yes' to 1, 'no' to 0, and anything else (including missing) to -1
        return phase_col.str.lower().map({'yes': 1, 'no': 0}).values

    def _extract_temperature(self, temp_str) -> float:
        """Extract numerical temperature values"""
        # dealing with Kelvin and Celsius
        temp_str = str(temp_str).strip()
        temp_str = temp_str.replace('＞', '>').replace('＜', '<').replace('～', '-').replace('–', '-').replace('—', '-')
        temp_str = temp_str.replace('℃', '°C')

        # Try to match patterns
        kelvin_two_sided_match = re.compile(r'(-?\d+\.?\d*)\s*[-~–—]\s*(-?\d+\.?\d*)\s*[Kk]').search(temp_str)
        kelvin_left_match = re.compile(r'[<]\s*(-?\d+\.?\d*)\s*[Kk]').search(temp_str)
        kelvin_right_match = re.compile(r'[>]\s*(-?\d+\.?\d*)\s*[Kk]').search(temp_str)

        celsius_two_sided_match = re.compile(r'(-?\d+\.?\d*)\s*(?:°C)?[-~–—]\s*(-?-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)
        celsius_left_pm_match = re.compile(r'[<≤]\s*(-?\d+\.?\d*)\s*(?:°C)??\s*±\s*(-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)
        celsius_left_match = re.compile(r'[<≤]\s*(-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)
        celsius_right_pm_match = re.compile(r'[>≥]\s*(-?\d+\.?\d*)\s*(?:°C)??\s*±\s*(-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)
        celsius_right_match = re.compile(r'[>≥]\s*(-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)
        celsius_exact_match = re.compile(r'(-?\d+\.?\d*)\s*(?:°C)?').search(temp_str)

        # Use if-elif structure instead of match-case for better compatibility
        if kelvin_two_sided_match:
            kelvin_low, kelvin_high = kelvin_two_sided_match.groups()
            # Ensure kelvin_low <= kelvin_high
            kelvin_low, kelvin_high = sorted([float(kelvin_low), float(kelvin_high)])
            celsius_low = kelvin_low - 273.15
            celsius_high = kelvin_high - 273.15
            return f"{celsius_low}, {celsius_high}"

        elif kelvin_left_match:
            celsius_high = float(kelvin_left_match.group(1)) - 273.15
            return f"0.0, {celsius_high}"

        elif kelvin_right_match:
            celsius_low = float(kelvin_right_match.group(1)) - 273.15
            return f"{celsius_low}, 99.99"

        elif celsius_two_sided_match:
            celsius_low, celsius_high = celsius_two_sided_match.groups()
            celsius_low, celsius_high = sorted([float(celsius_low), float(celsius_high)])
            return f"{celsius_low}, {celsius_high}"

        elif celsius_left_pm_match:
            celsius_mean, celsius_pm = celsius_left_pm_match.groups()
            mean_val = float(celsius_mean)
            pm_val = float(celsius_pm)
            return f"{0}, {mean_val + pm_val}"

        elif celsius_left_match:
            celsius_high = float(celsius_left_match.group(1))
            return f"{0.0}, {celsius_high}"

        elif celsius_right_pm_match:
            celsius_mean, celsius_pm = celsius_right_pm_match.groups()
            mean_val = float(celsius_mean)
            pm_val = float(celsius_pm)
            return f"{mean_val - pm_val}, 99.99"

        elif celsius_right_match:
            celsius_low = float(celsius_right_match.group(1))
            return f"{celsius_low}, 99.99"

        elif celsius_exact_match:
            celsius_exact = float(celsius_exact_match.group(1))
            return f"{celsius_exact}, {celsius_exact}"

        # Handle special string cases
        elif temp_str == "RT":
            return f"{25.0}, {25.0}"
        elif temp_str in ["freeze/thaw", "Freeze/thaw"]:
            return f"{0.0}, {25.0}"
        elif temp_str == "frozen":
            return f"{0.0}, {0.0}"
        elif temp_str == "warm by hand":
            return f"{25.0}, {37.0}"
        else:
            # If no match, return default room temperature bin
            logger.warning(f"Unrecognized temperature format: {temp_str}, set as missing value.")
            return

    def _extract_component_concentration(self, conc_str) -> float:
        """Extract numerical concentration values"""
        def map_one(conc_str):
            """Map single concentration string to float"""
            conc_str = str(conc_str).strip()
            conc_str = conc_str.replace('＞', '>').replace('＜', '<').replace('～', '-').replace('–', '-').replace('—', '-')
            # Handle different representations of 'µ' (micro sign)
            conc_str = conc_str.replace('μ', 'µ').replace('uM', 'µM')

            single_conc_match = re.compile(r'(\d+\.?\d*)(?:\-(\d+\.?\d*))?\s*(µM|mM|nM)(?:\s*\[[^\[\]]*\])?').search(conc_str)

            if single_conc_match:
                # Handle range like "10-20 µM"
                low, high, unit = single_conc_match.groups()
                mean = (float(low) + float(high)) / 2.0 if high else float(low)
                # Handle single value like "10 µM"
                match unit.lower():
                    case 'µm':
                        scale = 1
                    case 'mm':
                        scale = 1e+3
                    case 'nm':
                        scale = 1e-3
                    case _:
                        raise ValueError(f"Unknown concentration unit: {unit}")
                return  f"{mean*scale} µM"
            else:
                logger.warning(f"Unrecognized concentration format: {conc_str}, set as missing value.")
                return ''

        def map_many(conc_str):
            """Map protein ID to sequence using pre-built dict"""
            """Map multiple concentration strings to floats"""
            return ';'.join([map_one(x.strip()) for x in conc_str.split(';')])

        def map_conc(conc_str):
            """Map concentration string to float or list of floats"""
            if pd.isna(conc_str):
                # Handle missing component concentration
                return None
            if isinstance(conc_str, str) and ';' not in conc_str:
                # Single component concentration
                return map_one(conc_str)
            if isinstance(conc_str, str) and ';' in conc_str:
                # Handle multiple component concentrations
                return map_many(conc_str)

        return map_conc(conc_str)

    def _extract_ionic_strength(self, ionic_str) -> str:
        """Extract ionic strength values as standardized string (e.g., '150 mM')"""
        def map_one(ionic_str):
            ionic_str = str(ionic_str).strip()
            ionic_str = ionic_str.replace('＞', '>').replace('＜', '<').replace('～', '-').replace('–', '-').replace('—', '-')
            # Acceptable units: nM, mM, µM, M
            single_ionic_match = re.compile(r'(\d+\.?\d*)(?:\-(\d+\.?\d*))?\s*(nM|mM|µM|uM|μM|M)(?:\s*\[[^\[\]]*\])?').search(ionic_str)
            if single_ionic_match:
                low, high, unit = single_ionic_match.groups()
                mean = (float(low) + float(high)) / 2.0 if high else float(low)
                unit = unit.replace('uM', 'µM').replace('µM', 'μM')
                # Normalize to mM for output
                match unit.lower():
                    case 'nm':
                        scale = 1e-6
                    case 'μm':
                        scale = 1e-3
                    case 'mm':
                        scale = 1
                    case 'm':
                        scale = 1e+3
                    case _:
                        raise ValueError(f"Unknown concentration unit: {unit}")
                return f"{mean * scale} mM"
            else:
                logger.warning(f"Unrecognized ionic strength format: {ionic_str}, set as missing value.")
                return ''

        def map_many(ionic_str):
            return ';'.join([map_one(x.strip()) for x in ionic_str.split(';')])

        def map_ionic(ionic_str):
            if pd.isna(ionic_str):
                return None
            if isinstance(ionic_str, str) and ';' not in ionic_str:
                return map_one(ionic_str)
            if isinstance(ionic_str, str) and ';' in ionic_str:
                return map_many(ionic_str)

        return map_ionic(ionic_str)

    def _extract_buffer_pH(self, pH_str) -> str:
        """Extract buffer pH values"""
        pH_str = pH_str.strip().replace('～', '-').replace('–', '-').replace('−', '-')
        # Match pH values, including ranges like "pH 6.5-7.5" or "pH 7"
        pH_match = re.search(r'[Pp][Hh]\s*(?:\s+|at|\=)?\s*\(?(\d+\.?\d*)(?:\s*(?:-|to)\s*(\d+\.?\d*))?\)?', pH_str)
        if pH_match:
            low, high = pH_match.groups()
            mean = (float(low) + float(high)) / 2.0 if high else float(low)
            return f"pH {mean}"
        else:
            logger.warning(f"Unrecognized pH format: {pH_str}, set as missing value.")
            return ''

    def _extract_crowding_agent(self, crowding_str) -> float:
        """Extract numerical crowding agent values"""
        if isinstance(crowding_str, str) and crowding_str.strip() != '-':
            return 'Yes'
        if pd.isna(crowding_str):
            return np.nan
        if crowding_str == '-':
            return 'No'

    def _extract_systems_components(self, system_name) -> Tuple[int, bool, bool, bool]:
        """
        Extract system components type and return tuple of (num_proteins, has_rna, has_dna ,has_nucleic_acid)
        """
        num_proteins = int(re.search(r'protein\((\d+)\)', system_name).group(1))
        has_rna = 'RNA' in system_name
        has_dna = 'DNA' in system_name
        has_nucleic_acid = has_rna or has_dna

        return num_proteins, has_rna, has_dna, has_nucleic_acid

    def _validate_fix_raw_data(self, df_exp: pd.DataFrame, df_protein: pd.DataFrame):
        """Validate and fix raw data error and format inconsisitency before processing"""

        def fix_system(x):
            """Fix system names to match the expected format"""
            if x in self.system_list:
                return x
            else:
                y = x.replace("Protein", "protein").strip()
                y = y.replace("+  RNA", "+ RNA").replace("+  DNA", "+ DNA")
                logger.warning(f"fixing system name `{x}` to `{y}`")
                return y

        # unifiying the system names
        df_exp['Components type'] = df_exp['Components type'].map(fix_system)
        logger.info("Unifying system names in the dataset...")

        # doing some manual correction in the data for the labeling
        logger.info("Correcting manual errors in the dataset, Correcting the phase system for rows")
        error_rows_1 = [387, 1175, 1176, 1252, 2562, 2563, 2564, 2561, 4822, 4823, 4884, 4885, 5532, 5533,\
                        2605, 2606, 2903, 2904, 2905, 2906, 2907, 2908, 3013, 3897, 3898, 4566, 5243, 5706,\
                        5707, 5708, 5709, 5784, 5814, 5957, 6066] # Protein(1) should be Protein(2)
        error_rows_2 = [1177, 1178, 4824] # protein(1) + RNA should be protein(2) + RNA
        error_rows_3 = [5125, 5126, 5127 ,5128 ,5130, 5141, 5142] # Protein(1) should be Protein(6)
        df_exp.iloc[error_rows_1,2] = 'protein(2)'  # Correcting the error rows 1
        df_exp.iloc[error_rows_2,2] = 'protein(2) + RNA'  # Correcting the error rows 2
        df_exp.iloc[error_rows_3,2] = 'protein(6)'  # Correcting the error rows 3
        # df_exp.iloc[error_rows_1] # show the error rows 1
        # df_exp.iloc[error_rows_2] # show the error rows 2
        # df_exp.iloc[error_rows_3] # show the error rows 3
        logger.info(f"Corrected entries in the dataset: \n\
                    {error_rows_1} for protein(1) to protein(2)\n,\
                    {error_rows_2} for protein(1) + RNA to protein(2) + RNA, \n\
                    {error_rows_3} for protein(1) to protein(6).")

        assert set(df_exp['Components type'].unique().tolist()) == set(self.system_list), \
            f"Unexpected systems found in the dataset: {set(df_exp['Components type'].unique().tolist()) - set(self.system_list)}"
        # plot the system statistics after fixing the data
        return df_exp, df_protein

    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.system}, {len(self)})"

    def __getitem__(self, idx: int) -> ad.AnnData:
        """
        Get item by index from h5ad anndata file
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            AnnData object containing the selected data
        """
        adata = ad.read_h5ad(osp.join(self.embedding_dir, f'{self.system}.no_missing.h5ad'))
        return adata[idx]

    def __len__(self) -> int:
        """
        Get the length of the dataset
        Returns:
            int: Number of items in the dataset
        """
        df_system = pd.read_csv(osp.join(self.processed_dir, f'{self.system}.no_missing.txt'), sep='\t', index_col=0)
        return len(df_system)


# %%  ==================== Main ====================
if __name__ == "__main__":
    # raise RuntimeError("This script is not intended to be run directly. Use it as a module in your project.")
    # meta_config = create_meta_config(protein='esmc', dna_rna='dictionary', condition='droppler')
    # meta_config = create_meta_config(protein='mtdp', dna_rna='rnapsec_rna', condition='rnapsec_condition')
    # meta_config = create_meta_config(protein='rnapsec_protein', dna_rna='rnapsec_rna', condition='droppler')
    # meta_config = create_meta_config(protein='rnapsec_protein', dna_rna='dictionary', condition='rnapsec_condition_class')
    # meta_config = create_meta_config(protein='seq2onehot', dna_rna='dictionary', condition='rnapsec_condition_class')
    meta_config = create_meta_config(protein='seq2ind', dna_rna='dictionary', condition='droppler')

    pro1 = DB1Dataset(root='./data', name='DB1', system='protein(1)', embedding_config=meta_config)
    pro2 = DB1Dataset(root='./data', name='DB1', system='protein(2)', embedding_config=meta_config)
    pro3 = DB1Dataset(root='./data', name='DB1', system='protein(3)', embedding_config=meta_config)
    pro1dna = DB1Dataset(root='./data', name='DB1', system='protein(1) + DNA', embedding_config=meta_config)
    pro2dna = DB1Dataset(root='./data', name='DB1', system='protein(2) + DNA', embedding_config=meta_config)
    pro1rna = DB1Dataset(root='./data', name='DB1', system='protein(1) + RNA', embedding_config=meta_config)
    pro2rna = DB1Dataset(root='./data', name='DB1', system='protein(2) + RNA', embedding_config=meta_config)

