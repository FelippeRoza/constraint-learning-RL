import torch
import pickle
from torch.utils.data import Dataset


class HighwayDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, buffer_path):
        """
        Args:
            buffer_path (string): Path to the pickle file with experiences collected from simulation.
        """
        file_handler = open(buffer_path, 'rb')
        self.buffer = pickle.load(file_handler)

    def __len__(self):
        return len(self.buffer['action'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        for k, v in self.buffer.items():
            sample[k] = v[idx]

        return sample
