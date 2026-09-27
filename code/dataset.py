import numpy as np
import torch
from torch.utils.data import Dataset


class ESMNPZDataset(Dataset):
    """
    NPZ format:
      emb:    [N, L, H] float32
      mask:   [N, L]    int8/uint8/int64
      labels: [N]       int64
      ids:    [N]       object(str), optional
    """

    def __init__(self, npz_path, return_id=True):
        data = np.load(npz_path, allow_pickle=True)
        assert "emb" in data and "mask" in data and "labels" in data, (
            "NPZ must contain keys: emb, mask, labels"
        )

        self.emb = data["emb"].astype(np.float32)
        self.mask = data["mask"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)
        self.return_id = return_id

        if "ids" in data:
            self.ids = data["ids"]
        else:
            self.ids = np.asarray([f"sample_{i}" for i in range(len(self.labels))], dtype=object)

        assert len(self.emb) == len(self.mask) == len(self.labels) == len(self.ids)

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx):
        x_emb = torch.from_numpy(self.emb[idx])
        x_mask = torch.from_numpy(self.mask[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        sid = str(self.ids[idx])

        if self.return_id:
            return x_emb, x_mask, y, sid
        return x_emb, x_mask, y


def esm_collate_fn(batch):
    if len(batch[0]) == 4:
        x_emb, x_mask, y, sid = zip(*batch)
        return torch.stack(x_emb, dim=0), torch.stack(x_mask, dim=0), torch.stack(y, dim=0), list(sid)

    x_emb, x_mask, y = zip(*batch)
    return torch.stack(x_emb, dim=0), torch.stack(x_mask, dim=0), torch.stack(y, dim=0)
