import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_class import OpticalSimulationDatasetPD

def move_perm_map(data):
    perm_map = data.samples[0]['perm_map']

    # As perm_map is the same for all samples, store it separately
    new_data = OpticalSimulationDataset()
    new_data.samples = data.samples
    new_data.perm_map = perm_map

    for i in range(len(new_data)):
        new_data.samples[i].pop('perm_map', None)  # remove perm_map from each sample if it exists

    # Save the new dataset without perm_map in samples
    new_data.save_dataset("datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_256.pt")

# print(data.samples[0]['fields'].keys())
# print(data.samples[0]['fields']['Ex'].shape)

# Visualize a sample
# sample = data[510]
# plt.figure(figsize=(10, 5))
# plt.imshow(sample['fields']['Ex'].abs(), cmap='viridis')
# plt.colorbar()
# plt.title('Electric Field Ex Component')
# plt.show()

def convert_fields_size(data, factor):
    w, h = data.samples[0]['fields']['Ex'].shape
    new_w, new_h = w / factor, h / factor
    # We need new_w and new_h to be powers of 2
    w_pow = torch.log2(torch.tensor(new_w)).item()
    h_pow = torch.log2(torch.tensor(new_h)).item()
    print(f"Powers of 2: {w_pow}, {h_pow}")
    new_w = 2 ** round(w_pow)
    new_h = 2 ** round(h_pow)
    print(f"Resizing fields from ({w}, {h}) to ({new_w}, {new_h})")
    # Ask for confirmation
    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting.")
        return
    for i in tqdm(range(len(data))):
        for key in data.samples[i]['fields'].keys():
            field = data.samples[i]['fields'][key]
            field = field.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
            # Field is complex, so we need to resize real and imaginary parts separately
            field_real = F.interpolate(field.real, size=(new_w, new_h), mode='bilinear', align_corners=False)
            field_imag = F.interpolate(field.imag, size=(new_w, new_h), mode='bilinear', align_corners=False)
            field_resized = torch.complex(field_real, field_imag)
            data.samples[i]['fields'][key] = field_resized.squeeze(0).squeeze(0)  # remove batch and channel dimensions

    # Also resize perm_map
    perm_map = data.perm_map
    perm_map = perm_map.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
    perm_map_real = F.interpolate(perm_map.real, size=(new_w, new_h), mode='nearest')
    perm_map_imag = F.interpolate(perm_map.imag, size=(new_w, new_h), mode='nearest')
    perm_map_resized = torch.complex(perm_map_real, perm_map_imag)
    data.perm_map = perm_map_resized.squeeze(0).squeeze(0)

    data.save_dataset(f"datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_{new_w}.pt")

def convert_to_pandas(data):
    records = []
    for i in tqdm(range(len(data))):
        sample = data.samples[i]
        record = {
            'wavelength': sample['wavelength'],
            'theta': sample['theta'],
            'reflectance': sample['reflectance'],
            'transmittance': sample['transmittance'],
        }
        for key in sample['fields'].keys():
            record[key] = sample['fields'][key].numpy()
        records.append(record)
    records[0]['perm_map'] = data.perm_map.numpy()

    df = pd.DataFrame(records)
    df.to_pickle("datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pkl")
    print(f"DataFrame saved to datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pkl with {len(df)} samples.")
    print(df.head(2))



if __name__ == "__main__":
    data = OpticalSimulationDatasetPD(
        data_file="datasets/dataset_SinTiN_TORCWA_25_400_2000_25_0_70_4_64.pkl",
        fft=True,
        normalize=True
    )
    
    # Ex_fft = data.df.iloc[0]['Ex_fft']
    # Ex_fft_norm = data.df.iloc[0]['Ex_fft_norm']

    # plt.figure(figsize=(12, 5))
    # plt.subplot(2, 3, 1)
    # plt.imshow(Ex_fft.real, cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Real')
    # plt.subplot(2, 3, 2)
    # plt.imshow(Ex_fft.imag, cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Imaginary')
    # plt.subplot(2, 3, 3)
    # plt.imshow(Ex_fft.abs(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Magnitude')
    # plt.subplot(2, 3, 4)
    # plt.imshow(Ex_fft_norm.real, cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Norm Real')
    # plt.subplot(2, 3, 5)
    # plt.imshow(Ex_fft_norm.imag, cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Norm Imaginary')
    # plt.subplot(2, 3, 6)
    # plt.imshow(Ex_fft_norm.abs(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Ex FFT Norm Magnitude')
    # plt.tight_layout()
    # plt.show()


    # perm_map_fft = data.perm_map_fft
    # perm_map_fft_norm = data.perm_map_fft_norm

    # plt.figure(figsize=(12, 5)) 
    # plt.subplot(2, 3, 1)
    # plt.imshow(perm_map_fft.real, cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Real')
    # plt.subplot(2, 3, 2)
    # plt.imshow(perm_map_fft.imag, cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Imaginary')
    # plt.subplot(2, 3, 3)
    # plt.imshow(perm_map_fft.abs(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Magnitude')
    # plt.subplot(2, 3, 4)
    # plt.imshow(perm_map_fft_norm.real, cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Norm Real')
    # plt.subplot(2, 3, 5)
    # plt.imshow(perm_map_fft_norm.imag, cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Norm Imaginary')
    # plt.subplot(2, 3, 6)
    # plt.imshow(perm_map_fft_norm.abs(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Permittivity Map FFT Norm Magnitude')
    # plt.tight_layout()
    # plt.show()
