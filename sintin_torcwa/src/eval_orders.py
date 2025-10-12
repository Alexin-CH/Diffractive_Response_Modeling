# Python script: RCWA simulation of a layered metasurface using TORCWA
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torcwa

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1h20-1h30 on cpu for 50 wavelengths, 20 layers, 10 orders, 128x128y grid
# 20min on GPU for same parameters

sim_dtype = torch.complex64
geo_dtype = torch.float32

# Define lattice and geometry parameters (nm)
Lx, Ly = 1000.0, 1000.0    # Unit cell dimensions (periods) 
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = Lx
torcwa.rcwa_geo.Ly = Ly
torcwa.rcwa_geo.nx = 2**8    # Grid sampling (x)
torcwa.rcwa_geo.ny = 2**8    # Grid sampling (y)
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 1000.0

# Define materials
eps_air = 1.0
n_sub = 1.44
eps_sub = n_sub**2        # Substrate permittivity (SiO2)
eps_metal = 3.61 + 6.03j  # Metal (TiN) permittivity (from original code)

# Define patterned surface as sinusoidal corrugation of metal
amplitude = 55.0   # sin amplitude (nm)
period = 1000.0    # sin period (nm)
num_layers = 50    # number of discrete layers to approximate sinusoid
dz = (2*amplitude) / num_layers

# Pre-compute sine-wave heights
x = torcwa.rcwa_geo.x  # x-axis coordinates (nm)
# Create 2D coordinate mesh (rows=y, cols=x) for mask generation
X, Y = torch.meshgrid(torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, indexing='xy')
h = amplitude * torch.sin(2*np.pi * X / period) + amplitude  # height profile (0 to 2A)

def create_pattern_layer(z_thresh):
    """Return a mask (1 inside metal region, 0 outside) at height z_thresh."""
    return (h >= z_thresh).to(geo_dtype)  # binary mask

# Set incident wave parameters
inc_angle_deg = 30.0
inc_angle = inc_angle_deg * np.pi/180.0
azi_angle = 0.0

def eval_orders(order_list, angle_deg):
    """Compute reflection and transmission spectra over wavelengths at a given angle."""
    R_list = []
    T_list = []
    time_list = []
    tqdm_order = tqdm(order_list)
    for order in tqdm_order:
        t0 = time.time()
        wl = 1500
        tqdm_order.set_description(f"Simulating orders={order}")
        freq = 1.0 / wl
        sim = torcwa.rcwa(freq=freq, order=[order, order], L=[Lx, Ly], 
                          dtype=sim_dtype, device=device)
        sim.add_input_layer(eps=eps_air)            # superstrate (air)
        sim.add_output_layer(eps=eps_sub)           # substrate (SiO2)
        sim.set_incident_angle(inc_ang=angle_deg*np.pi/180, azi_ang=azi_angle)
        # Build patterned layers (metal vs substrate)
        for i in range(num_layers):
            z_mid = dz*(i+0.5) 
            mask = create_pattern_layer(z_mid)  # metal region mask at this layer
            layer_eps = mask * eps_metal + (1.0 - mask)
            # plt.imshow(layer_eps.real.cpu().numpy(), cmap='gray', aspect='auto')
            # plt.colorbar(label='Permittivity ε')
            # plt.title(f"Layer {i+1}/{num_layers} at z={z_mid:.1f} nm")
            # plt.xlabel('x (nm)')
            # plt.ylabel('y (nm)')
            # plt.tight_layout()
            # plt.show()
            tqdm_order.set_description(f"Adding layer {i+1}/{num_layers} for λ={wl:.1f} nm")
            sim.add_layer(thickness=dz, eps=layer_eps)
        # Add uniform metal layer below patterns
        sim.add_layer(thickness=50.0, eps=eps_metal)
        # Solve and get S-parameters for specular reflection/transmission (p-polarized)
        tqdm_order.set_description(f"Solving S-matrix for λ={wl:.1f} nm")
        sim.solve_global_smatrix()
        # Get Reflextion and Transmission coefficients for all polarizations
        R_pol = []
        T_pol = []
        for pol in ['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']:
            R0 = sim.S_parameters(orders=[0,0], direction='forward',
                                port='reflection', polarization=pol, ref_order=[0,0])
            T0 = sim.S_parameters(orders=[0,0], direction='forward',
                                port='transmission', polarization=pol, ref_order=[0,0])
            R_pol.append(abs(R0.cpu())**2)
            T_pol.append(abs(T0.cpu())**2)
        R_list.append(R_pol)
        T_list.append(T_pol)
        time_list.append(time.time() - t0)
    return np.array(R_list), np.array(T_list), np.array(time_list)

# Example: compute spectrum at 30° incidence from 800 to 2000 nm
orders = np.arange(0, 13) # Orders from 0 to 10
R_spec, T_spec, times = eval_orders(orders, inc_angle_deg)

# Find the polynomial fit for computation times
def polynomial_fit(x, y, degree):
    """Fit a polynomial of given degree to the data."""
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

quadratic_fit = polynomial_fit(orders, times, 2)
cubic_fit = polynomial_fit(orders, times, 3)
quartic_fit = polynomial_fit(orders, times, 4)

# Errors for polynomial fits
quadratic_error = np.mean((times - quadratic_fit(orders))**2)
cubic_error = np.mean((times - cubic_fit(orders))**2)
quartic_error = np.mean((times - quartic_fit(orders))**2)

# Plot computation times for each order
# plt.figure(figsize=(6,4))
# plt.plot(orders, times, marker='o', label='Computation Time')
# plt.plot(orders, quadratic_fit(orders), label=f"Quadratic Fit (error={quadratic_error:.5f})", linestyle='--')
# plt.plot(orders, cubic_fit(orders), label=f"Cubic Fit (error={cubic_error:.5f})", linestyle='--')
# plt.plot(orders, quartic_fit(orders), label=f"Quartic Fit (error={quartic_error:.5f})", linestyle='--')
# plt.title("Computation Time vs Order")
# plt.xlabel("Order")
# plt.ylabel("Time (seconds)")
# plt.legend()
# plt.grid()
# plt.show()

#exit(0)  # Exit after plotting computation times

# Plot reflection and transmission vs wavelength
for idx, pol in enumerate(['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']):
    plt.figure(figsize=(6,4))
    plt.plot(orders, R_spec[:, idx], label='Reflection (R)')
    plt.plot(orders, T_spec[:, idx], label='Transmission (T)')
    plt.title(f"Reflectance and Transmittance (θ={inc_angle_deg}°) at {pol} polarization")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid()
    plt.show()
