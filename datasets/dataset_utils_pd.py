import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class OpticalSimulationDatasetPD(Dataset):
    def __init__(self, data_file=None, transform=None, fft=False, normalize=False, logger=None):
        """
        Args:
            data_file: path to saved dataset file (.pkl).
            transform: optional callable to apply to each sample.
            logger: optional logger instance for logging messages.
        """
        self.transform = transform
        self.logger = logger
        self.df = None
        
        if data_file:
            self.load_dataset(data_file, fft=fft, normalize=normalize)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_dataset(self, output_file):
        """
        Save the dataset to a .pt file (PyTorch pickled dict).
        """
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if not output_file.endswith('.pkl'):
            output_file += ".pkl"

        self.df.to_pickle(output_file)
        
        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self

    def load_dataset(self, input_file, fft=False, normalize=True):
        """
        Load the dataset from a .pkl file.
        """
        if not input_file.endswith('.pkl'):
            raise ValueError("Output file must end with .pkl")

        self.df = pd.read_pickle(input_file)

        self.perm_map = self.df.iloc[0]['perm_map']
        
        if normalize:
            self.perm_map_mean = self.perm_map.mean()
            self.perm_map_std = self.perm_map.std()
            self.perm_map_norm = (self.perm_map - self.perm_map_mean) / self.perm_map_std

            fields_values = self.df[['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']].values.ravel()
            combined = np.concatenate([a.ravel() for a in fields_values])

            self.fields_mean = combined.mean()
            self.fields_std = combined.std()
            for field in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                self.df[f'{field}_norm'] = (self.df[field] - self.fields_mean) / self.fields_std

        if fft:
            self.perm_map_fft = np.fft.fftshift(np.fft.fft2(self.perm_map))

            if normalize:
                self.perm_map_fft_mean = self.perm_map_fft.mean()
                self.perm_map_fft_std = self.perm_map_fft.std()
                self.perm_map_fft_norm = (self.perm_map_fft - self.perm_map_fft_mean) / self.perm_map_fft_std

            for record in range(len(self.df)):
                for field in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                    fft = np.fft.fftshift(np.fft.fft2(self.df.at[record, field]))
                    self.df.at[record, f'{field}_fft'] = fft
            
            if normalize:
                fields_fft_values = self.df[[f'{field}_fft' for field in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']]].values.ravel()
                combined_fft = np.concatenate([a.ravel() for a in fields_fft_values])

                self.fields_fft_mean = combined_fft.mean()
                self.fields_fft_std = combined_fft.std()

                for field in fields:
                    self.df.loc[:, f'{field}_fft'] = [None] * N
                    self.df.loc[:, f'{field}_fft_norm'] = [None] * N

                for record in range(len(self.df)):
                    for field in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                        self.df.at[record, f'{field}_fft_norm'] = (self.df.at[record, f'{field}_fft'] - self.fields_fft_mean) / self.fields_fft_std            

        if self.logger:
            self.logger.info(f"Loaded {len(self.df)} samples from {input_file}")
        else:
            print(f"Loaded {len(self.df)} samples from {input_file}")
        return self
