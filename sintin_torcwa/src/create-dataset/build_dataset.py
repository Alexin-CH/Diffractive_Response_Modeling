import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class OpticalSimulationDataset(Dataset):
    def __init__(self, data_file=None, transform=None, logger=None):
        """
        Args:
            data_file: path to saved dataset file (.pt).
            transform: optional callable to apply to each sample.
            logger: optional logger instance for logging messages.
        """
        self.transform = transform
        self.samples = []
        self.logger = logger
        
        if data_file:
            self.load_dataset(data_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # torch DataLoader may pass a tensor idx
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def build_from_dir(self, keywords_list, dirname="data-outputs"):
        """
        Build the dataset by loading files from a directory that match the naming pattern.
        
        Args:
            nh: Numerical aperture or other identifier in the filename.
            discretization: Discretization level or other identifier in the filename.
            dirname: Directory containing the dataset files.
        """
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Directory {dirname} does not exist.")

        data_list = []
        n=0
        for file in os.listdir(dirname):
            if file.endswith('.pt') and all(keyword in file for keyword in keywords_list):
                n+=1
                input_file = os.path.join(dirname, file)
                sample = torch.load(input_file)
                data_list.append(sample)
        print(f"Done using {n} sample files")
        self.samples = data_list
        return self


    def save_dataset(self, output_file):
        """
        Save the dataset to a .pt file (PyTorch pickled dict).
        """
        parent_dir = os.path.dirname(output_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if not output_file.endswith('.pt'):
            output_file += ".pt"

        torch.save(self.samples, output_file)
        if self.logger:
            self.logger.info(f"Dataset saved to {output_file}")
        else:
            print(f"Dataset saved to {output_file}")
        return self

    def load_dataset(self, input_file):
        """
        Load the dataset from a .pt file.
        """
        if not input_file.endswith('.pt'):
            raise ValueError("Output file must end with .pt")

        data = torch.load(input_file)
        self.samples = data
        if self.logger:
            self.logger.info(f"Loaded {len(self.samples)} samples from {input_file}")
        else:
            print(f"Loaded {len(self.samples)} samples from {input_file}")
        return self


def main():
    dataset = OpticalSimulationDataset().build_from_dir(["sim", ""], dirname="data-outputs")
    title = f"dataset_sim_5Sin1000TiN_TORCWA_50_400_2000_50_0_70_3_256.pt"
    dataset.save_dataset("dataset/" + title)

if __name__ == "__main__":
    main()
