"""Small module to hold Dataset classes used by the notebooks.

Moving dataset classes out of the interactive notebook avoids multiprocessing
pickling/import issues on Windows (DataLoader workers use spawn).
"""

import time
from torch.utils.data import Dataset
import torch


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=100, delay_ms=0):
        """Synthetic dataset that can simulate I/O delay.

        Args:
            n_samples: number of samples
            n_features: number of features
            delay_ms: simulated I/O delay per sample in milliseconds
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.delay_ms = delay_ms

        # Pre-generate data (in practice, you'd load from disk)
        self.data = torch.randn(n_samples, n_features)
        self.labels = torch.randint(0, 10, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Simulate I/O delay
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        return self.data[idx], self.labels[idx]


__all__ = ["SyntheticDataset"]
