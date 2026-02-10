from pydantic import BaseModel, Field, field_validator
from typing import Optional


class MetaEmbeddingConfig(BaseModel):
    """Unified embedding configuration for protein, DNA/RNA, and condition."""

    protein_embedding: str = Field(
        default="esmc", description="Protein embedding model name. Allowed: 'esmc', 'mtdp', 'rnapsec_protein', 'seq2ind', 'seq2onehot'."
    )
    dna_rna_embedding: Optional[str] = Field(default="dictionary", description="DNA/RNA embedding model name. Allowed: 'rnapsec_rna', 'dictionary'.")
    condition_embedding: str = Field(
        default="droppler", description="Condition embedding model name. Allowed: 'droppler', 'rnapsec_condition', 'rnapsec_condition_class'."
    )

    @field_validator("protein_embedding")
    def validate_protein_embedding(cls, v):
        allowed = {"esmc", "mtdp", "rnapsec_protein", "seq2ind", "seq2onehot"}
        if v not in allowed:
            raise ValueError(f"Protein embedding '{v}' not allowed. Allowed: {allowed}")
        return v

    @field_validator("dna_rna_embedding")
    def validate_dna_rna_embedding(cls, v):
        allowed = {"rnapsec_rna", "dictionary"}
        if v not in allowed:
            raise ValueError(f"DNA/RNA embedding '{v}' not allowed. Allowed: {allowed}")
        return v

    @field_validator("condition_embedding")
    def validate_condition_embedding(cls, v):
        allowed = {"droppler", "rnapsec_condition", "rnapsec_condition_class"}
        if v not in allowed:
            raise ValueError(f"Condition embedding '{v}' not allowed. Allowed: {allowed}")
        return v


def create_meta_config(protein="mtdp", dna_rna="dictionary", condition="droppler"):
    """Create a unified embedding configuration."""
    return MetaEmbeddingConfig(protein_embedding=protein, dna_rna_embedding=dna_rna, condition_embedding=condition)
