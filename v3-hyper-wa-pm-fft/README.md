# v3: HyperNet wavelength-angle-permittivity map to fields prediction using FFT

## Description
Here the goal is to predict the EM fields in and around a 1D sine TiN structure given the wavelength, angle and the permittivity map of the structure.

But for this version, the we first compute the FFT of the permittivity map and the fields, and the model is trained to predict the FFT of the fields given the FFT of the permittivity map, wavelength and angle.

## Dataset
The dataset is generated using TORCWA, it contains 900 samples:
Range of parameters:
- Wavelength: 400nm - 2000nm
- Angle of incidence: 0 - 70 degrees

Fields are initially sampled on a 2D grid of size 256x500 (x,z).

## Model
The model is a HyperNet that takes as input the wavelength, angle and returns the weights of a CNN that takes as input the permittivity map and returns the fields in and around the structure.

