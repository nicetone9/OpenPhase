from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ConditionalProteinDataset(Dataset):
    def __init__(self, x, c, y):
        if x.ndim == 3 and x.shape[-1] == 21:
            x = torch.tensor(x).float() if not isinstance(x, torch.Tensor) else x.float()
            x = x.argmax(dim=-1)  # Now x is [N, L]
        else:
            x = torch.tensor(x).long() if not isinstance(x, torch.Tensor) else x.long()

        self.x = x  # [N, L] with Long type indices
        self.c = torch.tensor(c).float() if not isinstance(c, torch.Tensor) else c.float()
        self.y = torch.tensor(y).float() if not isinstance(y, torch.Tensor) else y.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx], self.y[idx]


def get_loader(x, c, y, batch_size=8, shuffle=False, num_workers=0, pin_memory=False):
    dataset = ConditionalProteinDataset(x, c, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader
     
