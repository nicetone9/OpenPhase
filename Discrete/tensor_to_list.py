import torch

IDX_TO_AA = {
    0: "-",   # padding / diffusion mask
    1: "A", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "K", 10: "L",
    11: "M", 12: "N", 13: "P", 14: "Q", 15: "R",
    16: "S", 17: "T", 18: "V", 19: "W", 20: "Y",
}

def tensor_to_sequence_list(tensor):
    """
    Accepts:
      - one-hot tensor [B, L, V]
      - index tensor   [B, L]

    Returns:
      - list[str] of length B
    """

    if tensor.dim() == 3:
        # one-hot ? indices
        idx = tensor.argmax(dim=-1)
    elif tensor.dim() == 2:
        idx = tensor
    else:
        raise ValueError(f"Expected tensor of shape [B,L] or [B,L,V], got {tensor.shape}")

    idx = idx.detach().cpu().tolist()

    seqs = []
    for seq in idx:
        s = "".join(IDX_TO_AA.get(i, "-") for i in seq)
        seqs.append(s)

    return seqs
