# Python script: RCWA simulation of a layered metasurface using TORCWA
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

import torcwa

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

sim_dtype = torch.complex64
geo_dtype = torch.float32

# Define lattice and geometry parameters (nm)
Lx, Ly = 1000.0, 1000.0    # Unit cell dimensions (periods)
L = [Lx, Ly]  # Lattice vector lengths
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = Lx
torcwa.rcwa_geo.Ly = Ly
torcwa.rcwa_geo.nx = 2**9    # Grid sampling (x)
torcwa.rcwa_geo.ny = 2**9    # Grid sampling (y)
torcwa.rcwa_geo.grid()

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

zmax = 1000
z = torch.linspace(-zmax, zmax, 1000, device=device)            # axe Z
nz = len(z)
ny = torcwa.rcwa_geo.ny

x_slice = Lx/2
x_idx   = torch.argmin(torch.abs(torcwa.rcwa_geo.x - x_slice)).item()

# Create 2D coordinate mesh (rows=y, cols=x) for mask generation
X, Y = torch.meshgrid(torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, indexing='xy')
h = amplitude * torch.sin(2*np.pi * X / period) + amplitude  # height profile (0 to 2A)

def create_pattern_layer(z_thresh):
    """Return a mask (1 inside metal region, 0 outside) at height z_thresh."""
    return (h >= z_thresh).to(sim_dtype)  # binary mask

# Set incident wave parameters
inc_angle_deg = 30.0
inc_angle = inc_angle_deg * np.pi/180.0
azi_angle = 0.0

def simulate_spectrum(wl_list, angle_deg):
    """Compute reflection and transmission spectra over wavelengths at a given angle."""
    R_list = []
    T_list = []
    tqdm_wl = tqdm(wl_list)
    for wl in tqdm_wl:
        tqdm_wl.set_description(f"Simulating λ={wl:.1f} nm")
        freq = 1.0 / wl
        sim = torcwa.rcwa(freq=freq, order=[0, 3], L=L, 
                          dtype=sim_dtype, device=device)
        perm_map = torch.zeros((nz, ny), dtype=sim_dtype, device=device)
        sim.add_input_layer(eps=eps_air)            # superstrate (air)
        sim.add_output_layer(eps=eps_sub)           # substrate (SiO2)

        mask_in = z < 0
        perm_map[mask_in, :] = eps_air
        sim.set_incident_angle(inc_ang=angle_deg*np.pi/180, azi_ang=azi_angle)

        # Build patterned layers (metal vs substrate)
        sim.add_layer(thickness=50.0, eps=eps_metal)
        cumul_z = 0.0
        for i in range(num_layers):
            z_mid = dz*(num_layers-i+0.5)
            mask = create_pattern_layer(z_mid)  # metal region mask at this layer
            layer_eps = mask * eps_metal + (1.0 - mask)
            sim.add_layer(thickness=dz, eps=layer_eps)
            idx = (z >= z_mid - dz/2) & (z < z_mid + dz/2)
            perm_map[idx, :] = layer_eps[x_idx, :]
            if False:
                plt.imshow(layer_eps.real.cpu().numpy(), cmap='gray', aspect='auto')
                plt.colorbar(label='Permittivity ε')
                plt.title(f"Layer {i+1}/{num_layers} at z={z_mid:.1f} nm")
                plt.xlabel('x (nm)')
                plt.ylabel('y (nm)')
                plt.tight_layout()
                plt.show()
            tqdm_wl.set_description(f"Adding layer {i+1}/{num_layers} for λ={wl:.1f} nm")

        # Add uniform metal layer below patterns
        idx = (z >= z_mid - dz/2) & (z < cumul_z +dz/2 + 50)
        perm_map[idx, :] = eps_metal

        # Add substrate layer
        mask_out = z >= cumul_z +dz/2 + 50
        perm_map[mask_out, :] = eps_sub

        # Plot permittivity map
        if False:
            plt.imshow(perm_map.real.cpu().numpy(), cmap='gray', aspect='auto')
            plt.colorbar(label='Permittivity ε')
            plt.title(f"Permittivity map for λ={wl:.1f} nm")
            plt.xlabel('x (nm)')
            plt.ylabel('z (nm)')
            plt.tight_layout()
            plt.show()

        

        # Solve and get S-parameters for specular reflection/transmission (p-polarized)
        tqdm_wl.set_description(f"Solving S-matrix for λ={wl:.1f} nm")
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
    return np.array(R_list), np.array(T_list), sim

# Example: compute spectrum at 30° incidence from 800 to 2000 nm
wavelengths = torch.linspace(1000.0, 1700.0, 100, dtype=geo_dtype)
R_spec, T_spec, sim = simulate_spectrum(wavelengths, inc_angle_deg)

wavelengths = wavelengths.cpu()  # Move to CPU for plotting
A_spec = 1.-R_spec - T_spec  # Absorption spectrum

# Plot reflection and transmission vs wavelength
if True:
    for idx, pol in enumerate(['xx', 'yy', 'pp', 'sp', 'ps', 'ss']):
        plt.figure(figsize=(6,4))
        plt.plot(wavelengths, T_spec[:, idx], label='Transmission (T)')
        plt.plot(wavelengths, R_spec[:, idx], label='Reflection (R)')
        #plt.plot(wavelengths, A_spec[:, idx], label='Absorption (A)')  
        plt.title(f"Reflectance and Transmittance (θ={inc_angle_deg}°) at {pol} polarization")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.grid()
        plt.show()

if False:
    # View XZ-plane fields and export
    sim.source_planewave(amplitude=[1.,0],direction='f')


    x_axis = torcwa.rcwa_geo.x.cpu()
    z_axis = z.cpu()

    t0 = time.time()
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_yz(torcwa.rcwa_geo.y,z,L[1]/2)
    print(f"Field calculation took {time.time()-t0:.2f} seconds")
    Enorm = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
    Hnorm = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)

    print(f"Fields at {1/sim.freq.abs().cpu():.2f}nm")

    fig, axes = plt.subplots(figsize=(10,12),nrows=2,ncols=4)
    im0 = axes[0,0].imshow(torch.transpose(Enorm,-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,0].set(title='E norm',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im1 = axes[0,1].imshow(torch.transpose(torch.abs(Ex),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,1].set(title='Ex abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im2 = axes[0,2].imshow(torch.transpose(torch.abs(Ey),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,2].set(title='Ey abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im3 = axes[0,3].imshow(torch.transpose(torch.abs(Ez),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,3].set(title='Ez abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    
    im4 = axes[1,0].imshow(torch.transpose(Hnorm,-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,0].set(title='H norm',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im5 = axes[1,1].imshow(torch.transpose(torch.abs(Hx),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,1].set(title='Hx abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im6 = axes[1,2].imshow(torch.transpose(torch.abs(Hy),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,2].set(title='Hy abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im7 = axes[1,3].imshow(torch.transpose(torch.abs(Hz),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,3].set(title='Hz abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    
    fig.colorbar(im0,ax=axes[0,0])
    fig.colorbar(im1,ax=axes[0,1])
    fig.colorbar(im2,ax=axes[0,2])
    fig.colorbar(im3,ax=axes[0,3])
    fig.colorbar(im4,ax=axes[1,0])
    fig.colorbar(im5,ax=axes[1,1])
    fig.colorbar(im6,ax=axes[1,2])
    fig.colorbar(im7,ax=axes[1,3])
    plt.tight_layout()
    plt.show()

