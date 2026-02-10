import os
import os.path as osp
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset

from meta_config import create_meta_config, MetaEmbeddingConfig
import gdown

from meta_embed import (get_condition_embeddings,
                                get_dna_rna_embeddings, get_protein_embeddings)

# %%
class DB2Dataset(Dataset):
    """
    Dataset for phase diagram data (DB2) in protein/nucleic acid systems. 
    This dataset is focused on the phase diagram applications and is designed to be simple and efficient.
    Subset IDR contains the continuous measurement of phase separation, while CoREST and RGG contain binary labels.

    - Supports subsets: 'CoREST', 'IDR', 'RGG'
    - Downloads, processes, and embeds data automatically
    - Uses AnnData for storage and access
    - Embedding methods are configurable via embedding_config

    Args:
        root (str): Root directory for dataset files.
        name (str): Dataset name.
        embedding_config (MetaEmbeddingConfig): Embedding configuration.
        subset_type (str): Experimental subset.
        download (bool): Download data if not found.

    Methods:
        download(): Download raw data.
        process(): Process raw Excel data.
        embed(): Embed and save data.
    """

    # Expected file name after download
    expected_file = "PS with Condition_Zhang Lab_v3.xlsx"

    def __init__(
        self,
        root: str = '.',
        name: str = 'DB2',
        embedding_config: MetaEmbeddingConfig = None,
        subset_type: str = 'CoREST',  # 'CoREST', 'IDR', 'RGG'
        download: bool = True,
    ):
        """
        Initialize PhaseDiagramDataset for phase diagram applications

        Args:
            root (str): Root directory where the dataset should be saved. Defaults to '.'.
            name (str): The name of the dataset. Defaults to 'DB2'.
            embedding_config (MetaEmbeddingConfig): Configuration for protein and condition embeddings.
            subset_type (str): The subset type (e.g., 'CoREST'). Defaults to 'CoREST'.
            download (bool): Whether to download data if not found. Defaults to True.
        """
        self.root = root
        self.name = name
        self.embedding_config = embedding_config
        self.subset_type = subset_type

        super().__init__()

        assert self.subset_type in self.subset_list, f"Subset `{self.subset_type}` is not in the list of available subsets: {self.subset_list}"

        self._create_directories()

        # Download data if needed
        if download and not self._check_raw_data_exists():
            self.download()

        # # Check if processed data exists, if not, process raw data
        if not self._check_processed_data_exists():
            self.process()

        # Check if embedding data exists, if not, embed the data
        if not self._check_embedding_data_exists():
            self.embed()

        logger.info(f"DB2 initialized with root: {self.root}, name: {self.name}")

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
    def subset_list(self) -> List[str]:
        """Return the list of available subsets in the dataset"""
        return ['CoREST', 'IDR', 'RGG']

    def download(self):
        """Download phase diagram data as XLSX from Google Drive using gdown"""
        if self._check_raw_data_exists():
            logger.info(f"File already exists in {self.raw_dir}, skip download.")
            return

        logger.info("Downloading phase diagram data from Google Drive using gdown...")

        # Create directories
        self._create_directories()
        file_path = osp.join(self.raw_dir, self.expected_file)

        try:
            # Google Drive file ID (replace with your actual file ID if needed)
            file_id = "1xMArSVic5by5Lr9WOrHzqastICAVcwwx"
            url = f"https://drive.google.com/uc?id={file_id}"

            logger.info(f"Downloading XLSX from {url}...")

            gdown.download(url, file_path, quiet=False)

            logger.info(f"Downloaded {self.expected_file} to {file_path}")
            logger.success("Download completed!")

        except Exception as e:
            logger.error(f"Error downloading {self.expected_file}: {e}")
            logger.warning("Please manually download the file from the Google Drive link")
            logger.warning(f"and save it as {file_path}")
            raise FileNotFoundError(f"Failed to download data file: {e}")

    def process(self):
        """
        Process the raw Excel data file and save the subset as a text file in the processed directory.
        """
        raw_file = osp.join(self.raw_dir, self.expected_file)

        logger.info(f"Processing raw data from {raw_file} for subset {self.subset_type}...")

        # Read Excel file
        df_subset = pd.read_excel(raw_file, sheet_name=self.subset_type)

        logger.info("Parsing phase separation labels")
        phase_label = df_subset['Phase_separation'] if self.subset_type != 'IDR' else df_subset['Intensity Fraction']

        logger.info("Parsing phase systems by compositions")
        phase_system = df_subset['Components_type']
        # logger.debug(f"Protein ID value counts:\n{df_exp['Protein ID'].value_counts()}")

        logger.info("Parsing protein sequences")
        phase_protein_id = df_subset['protID'].fillna('').str.replace(' ', '')
        phase_protein_seqs = df_subset['Protein Sequence'].replace(r'\s+', '', regex=True)

        logger.info("Parsing condition temperature")
        phase_temperature = df_subset['Temperature'].map({'RT': 25, '37˚C': 37})

        logger.info("Parsing condition component concentration")
        if self.subset_type == 'CoREST':
            phase_component_concentration = df_subset['Solute']
        else:
            phase_component_concentration = df_subset['Solute (Concentration)']

        logger.info("Parsing condition ionic strength")
        phase_ionic_strength = df_subset['Salt']

        logger.info("Parsing condition buffer pH")
        phase_buffer_ph = df_subset['Buffer']

        logger.info("Parsing condition crowding agent")
        phase_crowding_agent = df_subset['Crowding']

        logger.info("Parsing nucleic acid type")
        phase_nucleic_acid = df_subset['Nucleic_acid1']

        logger.info("Parsing nucleic acid seq")
        phase_nucleic_acid_seq = df_subset['Nucleic_acid1 sequence']

        df_parsed = pd.DataFrame({
            'Phase label': phase_label,
            'Phase system': phase_system,
            'Protein ID': phase_protein_id,
            'Protein sequences': phase_protein_seqs,
            'Protein (DNA/RNA) concentration': phase_component_concentration,
            'Nucleic acid': phase_nucleic_acid,
            'Nucleic acid sequence': phase_nucleic_acid_seq,
            'Temperature': phase_temperature,
            'Ionic strength': phase_ionic_strength,
            'Buffer pH': phase_buffer_ph,
            'Crowding agent': phase_crowding_agent,
        })
        df_parsed.index.name = self.subset_type

        df_parsed.to_csv(osp.join(self.processed_dir, f"{self.subset_type}.txt"), sep='\t', index=False)
        logger.info(f"Processed data saved to {osp.join(self.processed_dir, f'{self.subset_type}.txt')}")

    def embed(self):
        """
        Embed the data using the specified embedding configuration.
        This method should handle the embedding of protein and condition data.
        """
        df_subset = pd.read_csv(osp.join(self.processed_dir, f"{self.subset_type}.txt"), sep='\t')

        # Initialize embeddings dictionary
        protein_method = self.embedding_config.protein_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding
        condition_method = self.embedding_config.condition_embedding

        if self.subset_type == 'CoREST':
            # CoREST has no DNA/RNA AND 2 proteins
            phase_label = df_subset['Phase label'].map({'Yes': 1, 'No': 0})
            protein_1_embedding = get_protein_embeddings(df_subset['Protein sequences'].str.split(';', expand=True).iloc[:, 0].tolist(), protein_method)
            protein_2_embedding = get_protein_embeddings(df_subset['Protein sequences'].str.split(';', expand=True).iloc[:, 1].tolist(), protein_method)
            rna_embedding, rna_mapping = get_dna_rna_embeddings(df_subset['Nucleic acid sequence'], dna_rna_method)
            condition_embedding = get_condition_embeddings(df_subset, condition_method)

            embeddings_dict = {
                'y': phase_label,
                f'x_1_{protein_method}': protein_1_embedding,
                f'x_2_{protein_method}': protein_2_embedding,
                f'x_rna_{dna_rna_method}': rna_embedding,
                f'c_{condition_method}': condition_embedding
            }
            if rna_mapping is not None:
                embeddings_dict[f'rna_mapping_{dna_rna_method}'] = rna_mapping

        if self.subset_type == 'IDR':
            # IDR has no DNA/RNA AND 1 protein
            phase_label = df_subset['Phase label']
            protein_embedding = get_protein_embeddings(df_subset['Protein sequences'].tolist(), protein_method)
            condition_embedding = get_condition_embeddings(df_subset, condition_method)

            embeddings_dict = {
                'y': phase_label,
                f'x_{protein_method}': protein_embedding,
                f'c_{condition_method}': condition_embedding
            }

        if self.subset_type == 'RGG':
            # RGG has no DNA/RNA AND 1 protein
            phase_label = df_subset['Phase label'].map({'Yes': 1, 'No': 0})
            protein_embedding = get_protein_embeddings(df_subset['Protein sequences'].tolist(), protein_method)
            condition_embedding = get_condition_embeddings(df_subset, condition_method)

            embeddings_dict = {
                'y': phase_label,
                f'x_{protein_method}': protein_embedding,
                f'c_{condition_method}': condition_embedding
            }

        h5ad_file = osp.join(self.embedding_dir, f'{self.subset_type}.h5ad')
        h5ad_file_exists = osp.exists(h5ad_file)

        logger.info(f"Saving embeddings to new {h5ad_file}...") if not h5ad_file_exists else logger.info(f"Modifying embeddings to existing {h5ad_file}...")
        adata = ad.AnnData(X=df_subset.astype(str)) if not h5ad_file_exists else ad.read_h5ad(h5ad_file)

        for k, v in embeddings_dict.items():
            if k not in adata.obsm and k not in adata.uns:
                if 'mapping' not in k:
                    adata.obsm[k] = v.values
                else:
                    adata.uns[k] = v
            else:
                logger.warning(f"Key {k} already exists in AnnData object, skipping update.")
        adata.write_h5ad(h5ad_file)

        logger.info(f"Saved AnnData structure: {adata} at {h5ad_file}")


    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _check_raw_data_exists(self) -> bool:
        """
        Check if the expected raw data file exists in the raw directory.

        Returns:
            bool: True if the raw data file exists, False otherwise.
        """
        raw_file = osp.join(self.raw_dir, self.expected_file)
        raw_exist = osp.exists(raw_file)
        logger.info(f"Checking raw file in {self.raw_dir}... Exists: {raw_exist}")
        return raw_exist

    def _check_processed_data_exists(self) -> bool:
        """
        Check if the processed (parsed) data and system data exist.

        Returns:
            bool: True if processed data exists
        """
        subset_file = osp.join(self.processed_dir, f"{self.subset_type}.txt")
        subset_exist = osp.exists(subset_file)

        logger.info(f"Checking processed files in {self.processed_dir}...")
        logger.info(f"Subset file exists: {subset_exist}")

        return subset_exist

    def _check_embedding_data_exists(self) -> bool:
        """
        Check if the embedding data file exists for the current subset.

        Returns:
            bool: True if the embedding data file exists, False otherwise.
        """
        # tensor_file = osp.join(self.embedding_dir, f"{self.subset_type}.pt")
        protein_method = self.embedding_config.protein_embedding
        condition_method = self.embedding_config.condition_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding

        h5ad_file = osp.join(self.embedding_dir, f"{self.subset_type}.h5ad")
        embedding_file_exist = osp.exists(h5ad_file)
        embedding_data_exist = False

        if embedding_file_exist:
            adata = ad.read_h5ad(h5ad_file)
            obsm_keys = adata.obsm.keys()

            protein_key_exist = any(protein_method in k for k in obsm_keys if k.startswith('x_'))
            condition_key_exist = any(condition_method in k for k in obsm_keys if k.startswith('c_'))
            dna_rna_key_exist = any(dna_rna_method in k for k in obsm_keys if k.startswith('x_dna_') or k.startswith('x_rna_'))
            embedding_data_exist = protein_key_exist and condition_key_exist and dna_rna_key_exist

            logger.debug("Checking embedding data keys in h5ad file...")
            logger.debug(f"Protein embedding key exists: {protein_key_exist} ({protein_method})")
            logger.debug(f"Condition embedding key exists: {condition_key_exist} ({condition_method})")
            logger.debug(f"DNA/RNA embedding key exists: {dna_rna_key_exist} ({dna_rna_method}) Default as True when no nucleic acid in the system:")

        return embedding_file_exist and embedding_data_exist

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.subset_type}, {len(self)})"

    def __getitem__(self, idx: int) -> ad.AnnData:
        """
        Get item by index
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            ad.AnnData: AnnData object containing the data for the specified index
        """
        adata_file = osp.join(self.embedding_dir, f"{self.subset_type}.h5ad")
        adata = ad.read_h5ad(adata_file)

        # Return the data at the specified index
        return adata[idx]

    def __len__(self) -> int:
        """
        Get the length of the dataset
        Returns:
            int: Number of items in the dataset
        """
        df_subset = pd.read_csv(osp.join(self.processed_dir, f'{self.subset_type}.txt'), sep='\t')
        return len(df_subset)

# %% # ==================== Main ====================


if __name__ == "__main__":
    # raise RuntimeError("This script is not intended to be run directly. Use it as a module in your project.")
    meta_config = create_meta_config(protein='rnapsec_protein', dna_rna='rnapsec_rna', condition='rnapsec_condition')

    corest = DB2Dataset(root='./data', name='DB2', embedding_config=meta_config, subset_type='CoREST', download=True)
    idr = DB2Dataset(root='./data', name='DB2', embedding_config=meta_config, subset_type='IDR', download=True)
    rgg = DB2Dataset(root='./data', name='DB2', embedding_config=meta_config, subset_type='RGG', download=True)



    # self = corest
    # db1 = pd.read_csv('../DB1/processed/exp_parsed.txt', sep='\t', index_col=0)
    # db1_pro = pd.read_csv('../DB1/processed/exp_protein.txt', sep='\t', index_col=0)
    # db2_corest = pd.read_csv('../DB2/processed/CoREST.txt', sep='\t', index_col=0)
    # db2_idr = pd.read_csv('../DB2/processed/IDR.txt', sep='\t', index_col=0)
    # db2_rgg = pd.read_csv('../DB2/processed/RGG.txt', sep='\t', index_col=0)
