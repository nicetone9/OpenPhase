IDX_TO_AA = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"  # 20 AAs + 'X' for unknown
]

def tensor_to_sequence_list(one_hot_tensor, mask=None):
    """
    Converts a one-hot encoded tensor of shape [B, L, 21] to a list of amino acid sequences.
    """
    B, L, _ = one_hot_tensor.shape
    idx_tensor = one_hot_tensor.argmax(dim=-1)  # [B, L]

    sequences = []
    for i in range(B):
        seq = ""
        for j in range(L):
            if mask is not None and mask[i, j]:
                seq += "X"  # or "-" or "?" for masked positions
            else:
                seq += IDX_TO_AA[idx_tensor[i, j].item()]
        sequences.append(seq)
    return sequences

