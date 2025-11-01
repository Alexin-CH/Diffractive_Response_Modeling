import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

class OpticalSimulationDatasetPD(Dataset):
    def __init__(self, data_file=None, transform=None, logger=None, device='cpu', dtype=torch.complex64):
        """
        Args:
            data_file: path to saved dataset file (.pkl).
            transform: optional callable to apply to each sample.
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
            self.load_dataset(data_file, fft=fft, normalize=normalize)

    def __len__(self):
        return 0 if self.df is None else len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx].copy().to_dict()

        if self.transform:
            sample = self.transform(sample)
        return sample

    def save_dataset(self, output_file):
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if output_file.endswith('.pkl'):
            self.df.to_pickle(output_file)
        elif output_file.endswith('.csv'):
            self.df.to_csv(output_file, index=False)
        else:
            raise ValueError("Output file must end with .pkl or .csv")

        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self


    def load(self, input_file):
        if input_file.endswith('.pkl'):
            self.df = pd.read_pickle(input_file)
        elif input_file.endswith('.csv'):
            self.df = pd.read_csv(input_file)
        else:
            raise ValueError("Input file must end with .pkl or .csv")
        return self

    def build_from_dir(self, dirname, keywords_list=[], anti_keywords_list=[]):
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Directory {dirname} does not exist.")

        data_list = []
        n=0

        tqdm_listdir = tqdm(os.listdir(dirname), desc="Loading dataset files")
        for file in tqdm_listdir:
            kw = all(keyword in file for keyword in keywords_list)
            akw = any(anti_keyword in file for anti_keyword in anti_keywords_list)
            if kw and not akw:
                if file.endswith('.json'):
                    input_file = os.path.join(dirname, file)
                    sample_df = pd.read_json(input_file)
                    data_list.append(sample_df)
                    n+=1
        print(f"Done using {n} sample files")
        print("Concatenating data to a single DataFrame...", end=' ')
        self.df = pd.concat(data_list, ignore_index=True)
        print("Done.")
        return self

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = OpticalSimulationDatasetPD(device=device)
    dataset.build_from_dir(
        dirname=f"{current_dir}/create-dataset/samples",
        keywords_list=["sample", "nh30.dis256", ".json"],
        anti_keywords_list=["map"]
    )

    print("Dataset info:")
    print(dataset.df.info())
    print("Dataset description:")
    print(dataset.df.describe())
    print("First 5 entries:")
    print(dataset.df.head())

    df_no_nan = dataset.df.dropna()
    print("NaN diff:", len(dataset.df) - len(df_no_nan))

    local_output_file = f"{current_dir}/dataset_STT_{len(dataset.df)}_30nh.local.csv"
    dataset.save_dataset(local_output_file)

    output_file = f"{current_dir}/dataset_STT_30nh.csv"
    dataset.save_dataset(output_file)
    