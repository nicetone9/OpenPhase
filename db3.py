import os
import os.path as osp

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from meta_config import MetaEmbeddingConfig
import requests
import re

from meta_embed import (get_condition_embeddings, get_dna_rna_embeddings, get_protein_embeddings)

# %%
class DB3Dataset(Dataset):
    """
    Dataset for phase diagram data (DB3) in protein + RNA systems.
    The DB3 dataset is a preprocessed resource derived from RNAPSEC (Chin et al. 2024),
    with experimental information and phase component data manually extracted from public literature
    in RNAPhaSep (Zhu et al. 2022). DB3 focuses on the phase behavior of protein-RNA systems,
    providing unified numeric representations to enable prediction of phase boundaries and
    condition-dependent behaviors across diverse protein-RNA combinations.

    Args:
        root (str): Root directory for dataset files.
        name (str): Dataset name.
        embedding_config (MetaEmbeddingConfig): Embedding configuration.
        download (bool): Download data if not found.

    Methods:
        download(): Download raw data.
        process(): Process raw Excel data.
        embed(): Embed and save data.
    """
    # Expected file name after download
    expected_file = "rnapsec.xlsx"

    def __init__(
        self,
        root: str = '.',
        name: str = 'DB3',
        embedding_config: MetaEmbeddingConfig = None,
        download: bool = True,
    ):
        """
        Initializes the PhaseDiagramDataset for phase diagram applications.

        Parameters:
            root (str): The root directory where the dataset will be stored. Defaults to '.'.
            name (str): The name identifier for the dataset. Defaults to 'DB3'.
            embedding_config (MetaEmbeddingConfig, optional): Configuration object for protein and condition embeddings.
            download (bool): If True, downloads the dataset if it is not already present. Defaults to True.

        This constructor sets up the dataset directory structure, downloads raw data if necessary, processes the data if not already processed, and generates embeddings if they do not exist.

        """
        self.root = root
        self.name = name
        self.embedding_config = embedding_config

        super().__init__()

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

        logger.info(f"DB3 initialized with root: {self.root}, name: {self.name}")

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

    def download(self):
        """Download RNAPSEC data XLSX from GitHub if not present."""
        if self._check_raw_data_exists():
            logger.info(f"File already exists in {self.raw_dir}, skip download.")
            return

        logger.info("Downloading RNAPSEC data from GitHub...")
        file_path = osp.join(self.raw_dir, self.expected_file)

        try:
            # Direct download link for the raw XLSX file from GitHub
            url = "https://github.com/ycu-iil/RNAPSEC/raw/main/data/rnapsec.xlsx"

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {self.expected_file} to {file_path}")
            logger.success("Download completed!")

        except Exception as e:
            logger.error(f"Error downloading {self.expected_file}: {e}")
            logger.warning(f"Please manually download the file from {url}")
            logger.warning(f"and save it as {file_path}")
            raise FileNotFoundError(f"Failed to download data file: {e}")

    def process(self):
        """
        Process the raw Excel data file and save the subset as a text file in the processed directory.
        """
        raw_file = osp.join(self.raw_dir, self.expected_file)

        logger.info(f"Processing raw data from {raw_file}")

        # Read Excel file
        df_rnapsec = pd.read_excel(raw_file)
        # df_rnapsec['morphology_add'].value_counts()
        # df_rnapsec['solute_concentration'].str.split(';', expand=True)
        # df_rnapsec[df_rnapsec['components_type'] == 'RNA + proteins']


        logger.info("Parsing phase separation labels")
        phase_label = df_rnapsec['morphology_add'] 
        # phase_mapping = {"liquid": "Yes", "solid": "Yes", "gel": "Yes", "solute": "No"}
        # phase_label.value_counts()

        logger.info("Parsing phase systems by compositions")
        phase_system = df_rnapsec['components_type'].map({'RNA + protein': 'protein(1) + RNA', 'RNA + proteins': 'protein(1) + RNA',})

        logger.info("Parsing protein IDs")
        phase_protein_id = df_rnapsec['rnaphasep_Uniprot ID']


        logger.info("Parsing protein sequences")
        phase_protein_seqs = self._extract_protein_sequences(df_rnapsec)

        # phase_protein_seqs.iloc[1104:1108]
        # df_rnapsec[df_rnapsec.rpsid=='RNAPS0000434']
        # phase_protein_seqs[phase_protein_seqs.str.contains("[0-9]", na=True)].unique()

        logger.info("Parsing condition temperature")
        phase_temperature = (df_rnapsec['temperature'].replace({'RT': '25'}).astype(str).str.extract(r'(\d+\.?\d*)')[0])


        logger.info("Parsing condition component concentration")
        phase_component_concentration = self._extract_component_concentration(df_rnapsec)

        logger.info("Parsing condition ionic strength")
        # Extract all numbers from rnaphasep_salt_concentration and sum them for each row
        phase_ionic_strength = df_rnapsec['rnaphasep_salt_concentration'].apply(self._extract_ionic_strength)
        phase_ionic_strength = phase_ionic_strength.str.split(';', expand=True)\
                                .apply(lambda x: f"{x.str.replace('mM', '').astype(float).sum()} mM" if pd.notna(x).any() else np.nan, axis=1)

        logger.info("Parsing condition buffer pH")
        phase_buffer_ph = 'pH ' + df_rnapsec['pH'].astype(str)

        logger.info("Parsing condition crowding agent")
        phase_crowding_agent = df_rnapsec['rnaphasep_other_requirement'].apply(lambda x: 'Yes' if 'crowding' in x else 'No')

        logger.info("Parsing nucleic acid type")
        phase_nucleic_acid = df_rnapsec['rnaphasep_rnas']

        logger.info("Parsing nucleic acid seq")
        phase_nucleic_acid_seq = df_rnapsec['rnaphasep_rna_sequence'].str.replace(';|-','').replace(';|-','').apply(lambda x: x if x!='-' else np.nan)


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

        df_no_missing = df_parsed[df_parsed.notna().all(axis=1)]

        logger.info(f"Processed data saved to {osp.join(self.processed_dir, 'rnapsec_1514.txt')}")
        df_parsed.to_csv(osp.join(self.processed_dir, "rnapsec_1514.txt"), sep='\t', index=False)
        df_no_missing.to_csv(osp.join(self.processed_dir, "rnapsec_no_missing_1298.txt"), sep='\t', index=False)

    def embed(self):
        """
        Embed the data using the specified embedding configuration.
        This method should handle the embedding of protein and condition data.
        """
        df_no_missing = pd.read_csv(osp.join(self.processed_dir, 'rnapsec_no_missing_1298.txt'), sep='\t')

        protein_method = self.embedding_config.protein_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding
        condition_method = self.embedding_config.condition_embedding

        # RNAPSEC's phase label is by type
        phase_mapping = {"liquid": 1, "solid": 1, "gel": 1, "solute": 0}
        phase_label = df_no_missing['Phase label'].map(phase_mapping)

        protein_embedding = get_protein_embeddings(df_no_missing['Protein sequences'].tolist(), protein_method)  # esmc mtdp rnapsec_protein
        rna_embedding, rna_mapping = get_dna_rna_embeddings(df_no_missing['Nucleic acid sequence'], dna_rna_method)
        condition_embedding = get_condition_embeddings(df_no_missing, condition_method)

        embeddings_dict = {
            'y': phase_label,
            f'x_{protein_method}': protein_embedding,
            f'x_rna_{dna_rna_method}': rna_embedding,
            f'c_{condition_method}': condition_embedding
        }
        if rna_mapping is not None:
            embeddings_dict[f'rna_mapping_{dna_rna_method}'] = rna_mapping


        h5ad_file = osp.join(self.embedding_dir, "rnapsec_no_missing_1298.h5ad")
        h5ad_file_exists = osp.exists(h5ad_file)
        logger.info(f"Saving embeddings to new {h5ad_file}...") if not h5ad_file_exists else logger.info(f"Modifying embeddings to existing {h5ad_file}...")
        adata = ad.AnnData(X=df_no_missing.astype(str)) if not h5ad_file_exists else ad.read_h5ad(h5ad_file)

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
        parsed_file = osp.join(self.processed_dir, "rnapsec_1514.txt")
        no_missing_file = osp.join(self.processed_dir, "rnapsec_no_missing_1298.txt")
        parsed_file_exist = osp.exists(parsed_file)
        no_missing_file_exist = osp.exists(no_missing_file)

        logger.info(f"Checking processed files in {self.processed_dir}...")
        logger.info(f"Parsed file exists: {parsed_file_exist}")
        logger.info(f"No_missing file exists: {no_missing_file_exist}")

        return parsed_file_exist and no_missing_file_exist

    def _check_embedding_data_exists(self) -> bool:
        """
        Check if the embedding data file exists for the current subset.

        Returns:
            bool: True if the embedding data file exists, False otherwise.
        """

        protein_method = self.embedding_config.protein_embedding
        condition_method = self.embedding_config.condition_embedding
        dna_rna_method = self.embedding_config.dna_rna_embedding

        h5ad_file = osp.join(self.embedding_dir, "rnapsec_no_missing_1298.h5ad")
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

    def _extract_protein_sequences(self, df_rnapsec: pd.DataFrame) -> pd.Series:
        """
        Extract and clean protein sequences from the DataFrame using a specific procedure.
        Assumes the protein sequence is in the 'protein_sequence' column.
        """
        df_rnapsec = df_rnapsec.copy()
        df_rnapsec["aa"] = np.nan

        # If no "|" in sequence, use as is
        mask_no_pipe = ~df_rnapsec.protein_sequence.str.contains("\|", na=False)
        df_rnapsec.loc[mask_no_pipe, "aa"] = df_rnapsec.loc[mask_no_pipe, "protein_sequence"]

        # If "|" present, extract after first newline and clean
        mask_pipe = df_rnapsec.protein_sequence.str.contains("\|", na=False)
        df_rnapsec.loc[mask_pipe, "aa"] = (
            df_rnapsec.loc[mask_pipe, "protein_sequence"]
            .str.split("\n", n=1, expand=True)[1]
            .str.replace("\n|\||;", "", regex=True)
        )

        # Further cleaning
        df_rnapsec["aa"] = df_rnapsec["aa"].str.replace("\n", "", regex=False)
        df_rnapsec["aa"] = df_rnapsec["aa"].str.replace("|", "", regex=False)
        df_rnapsec["aa"] = df_rnapsec["aa"].str.replace(";", "", regex=False)
        df_rnapsec["aa"] = df_rnapsec["aa"].str.replace("(^[A-Z])", "", regex=False)
        df_rnapsec["aa"] = df_rnapsec["aa"].str.replace(
            ">spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1",
            "",
            regex=False,
        )
        logger.debug("#Edit fusion protein >spQ99496RING2_HUMAN E3 ubiquitin-protein ligase RING2 OS=Homo sapiens OX=9606 GN=RNF2 PE=1 SV=1")

        df_rnapsec.loc[1104:1108,"aa"] = "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPINTNSSPDDQIGYYRRATRRIRGGDGKMKDLSPRWYFYYLGTGPEAGLPYGANKDGIIWVATEGALNTPKDHIGTRNPANNAAIVLQLPQGTTLPKGFYAEGSRGGSQASSRSSSRSRNSSRNSTPGSSRGTSPARMAGNGGDAALALLLLDRLNQLESKMSGKGQQQQGQTVTKKSAAEASKKPRQKRTATKAYNVTQAFGRRGPEQTQGNFGDQELIRQGTDYKHWPQIAQFAPSASAFFGMSRIGMEVTPSGTWLTYTGAIKLDDKDPNFKDQVILLNKHIDAYKTFPPTEPKKDKKKKADETQALPQRQKKQQTVTLLPAADLDDFSKQLQQSMSSADSTQA"
        logger.debug("Fix the residue index and retrive the sequence for index 1104 1105 1106 1107")

        # Filter for valid sequences
        # df_rnapsec = df_rnapsec[df_rnapsec.aa.str.contains("[A-Z][A-Z]", na=False)]
        # df_rnapsec = df_rnapsec[df_rnapsec.aa.notna()]

        # Check for non-single letter codes
        # print(df_rnapsec[df_rnapsec.aa.str.contains("[0-9]", na=True)].aa.unique())
        # assert (
        #     df_rnapsec[df_rnapsec.aa.str.contains("[0-9]", na=True)].shape[0] == 0
        # ), "protein seq contains non single letter"

        return df_rnapsec["aa"]

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
            return ';'.join([map_one(x.strip()) for x in ionic_str.split(',')])

        def map_ionic(ionic_str):
            ionic_str = re.sub(r'\([^)]*\)', '', ionic_str)
            if pd.isna(ionic_str) or ionic_str == '-':
                return None
            if isinstance(ionic_str, str) and ',' not in ionic_str:
                return map_one(ionic_str)
            if isinstance(ionic_str, str) and ',' in ionic_str:
                return map_many(ionic_str)

        return map_ionic(ionic_str)

    def _extract_component_concentration(self, df_rnapsec: pd.DataFrame) -> pd.Series:
        """
        Extracts and unifies protein and RNA concentrations from a DataFrame, returning them as a formatted string in µM units.
        """
        def unify_unit(value, unit):
            unit_new = str(unit).strip().replace('uM', 'µM').replace('μM', 'µM').replace('miu M', 'µM')

            if unit_new.lower() in ['µm']:
                value = value if not pd.isna(value) else np.nan
            elif unit_new.lower() in ['mm']:
                value = value * 1000 if not pd.isna(value) else value
            elif unit_new.lower() in ['nm']:
                value = value * 0.001 if not pd.isna(value) else value
            elif unit_new.lower() in ['m']:
                value = value * 1e6 if not pd.isna(value) else value
            else:
                logger.warning(f"Unknown unit {unit}, keeping original value.")

            return f"{value} µM" if not pd.isna(value) else np.nan

        # Apply unit unification
        protein_conc_unified = df_rnapsec.apply(lambda row: unify_unit(row['protein_conc'], row['protein_unit']), axis=1)
        rna_conc_unified = df_rnapsec.apply(lambda row: unify_unit(row['rna_conc'], row['rna_unit']), axis=1)

        return protein_conc_unified + ';' + rna_conc_unified

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __getitem__(self, idx: int) -> ad.AnnData:
        """
        Get item by index
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            ad.AnnData: AnnData object containing the data for the specified index
        """
        adata_file = osp.join(self.embedding_dir, "rnapsec_no_missing_1298.h5ad")
        adata = ad.read_h5ad(adata_file)

        # Return the data at the specified index
        return adata[idx]

    def __len__(self) -> int:
        """
        Get the length of the dataset
        Returns:
            int: Number of items in the dataset
        """
        df_rnapsec = pd.read_csv(osp.join(self.processed_dir, "rnapsec_no_missing_1298.txt"), sep='\t')
        return len(df_rnapsec)

# %%
# ==================== Main ====================
if __name__ == "__main__":
    raise RuntimeError("This script is not intended to be run directly. Use it as a module in your project.")
    # meta_config = create_meta_config(protein='rnapsec_protein', dna_rna='rnapsec_rna', condition='rnapsec_condition')
    # rnapsec = DB3Dataset(root='..', name='DB3', embedding_config=meta_config, download=True)
