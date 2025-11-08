import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class OpticalSimulationDatasetPD(Dataset):
    def __init__(self, data_file=None, transform=None, normalize=False, logger=None, device='cpu', dtype=torch.complex64):
        """
        Args:
            data_file: path to saved dataset file (.pkl).
            transform: optional callable to apply to each sample.
            fft: compute FFTs for perm_map and field arrays when loading.
            normalize: whether to compute & store normalization stats and normalized arrays.
            logger: optional logger instance.
            device: device for tensors ('cpu' or 'cuda').
            dtype: torch dtype for numeric tensors.
        """
        self.transform = transform
        self.logger = logger
        self.df = None
        self.device = torch.device(device)
        self.dtype = dtype

        if data_file:
            self.load_dataset(data_file, normalize=normalize)

    def __len__(self):
        return 0 if self.df is None else len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx].copy().to_dict()

        # convert any torch tensors back to tensors (they are stored as torch tensors already)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_dataset(self, output_file):
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if not output_file.endswith('.csv'):
            output_file += ".csv"

        # DataFrame may contain torch tensors; pandas will pickle them fine.
        self.df.to_pickle(output_file)

        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self

    def load_dataset(self, input_file, normalize=False):
        if not input_file.endswith('.csv'):
            raise ValueError("Input file must end with .csv")

        self.df = pd.read_csv(input_file).reset_index()

        # Transmissions and reflections are stored as dict 'real', 'imag'. Convert to complex.
        for col in tqdm(self.df.columns, desc="Loading dataset features"):
            if col.startswith('forward') or col.startswith('backward'):
                self.df[col] = self.df[col].apply(lambda x: complex(eval(x)['real'], eval(x)['imag']))

        if self.logger:
            self.logger.info(f"Loaded {len(self.df)} samples from {input_file}")
        else:
            print(f"Loaded {len(self.df)} samples from {input_file}")
        return self
