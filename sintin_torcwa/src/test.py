#!/usr/bin/env python3
# RCWA simulation of a layered metasurface using TORCWA
import os
import time

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torcwa

# ── 1) Device et dtypes ───────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

sim_dtype = torch.complex64
geo_dtype = torch.float32

# ── 2) Définition de la géométrie ────────────────────────────────────────────
Lx, Ly = 1000.0, 1000.0   # dimensions de la maille (nm)
L = [Lx, Ly]

torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = Lx
torcwa.rcwa_geo.Ly = Ly
torcwa.rcwa_geo.nx = 2**9  # grille en x
torcwa.rcwa_geo.ny = 2**9  # grille en y
torcwa.rcwa_geo.grid()

# ── 3) Matériaux ─────────────────────────────────────────────────────────────
eps_air   = torch.tensor(1.0, dtype=sim_dtype, device=device)
n_sub     = 1.44
eps_sub   = torch.tensor(n_sub**2, dtype=sim_dtype, device=device)
eps_metal = torch.tensor(3.61 + 6.03j, dtype=sim_dtype, device=device)

# ── 4) Paramètres de la corrugation sinusoïdale ─────────────────────────────
amplitude  = 55.0    # nm
period     = 1000.0  # nm
num_layers = 50
dz         = (2*amplitude) / num_layers  # épaisseur de chaque tranche

# ── 5) Axe z et coupe en x ──────────────────────────────────────────────────
zmax = 2000.0
z    = torch.linspace(-zmax, zmax, 1000, device=device)  # axe Z
nz   = z.numel()
ny   = torcwa.rcwa_geo.ny

x_slice = Lx/2
x_idx   = torch.argmin(torch.abs(torcwa.rcwa_geo.x - x_slice)).item()

# Pré-calcul du profil h(x,y)
X, Y = torch.meshgrid(torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, indexing='xy')
h = amplitude * torch.sin(2*np.pi * X / period) + amplitude

def create_pattern_layer(z_thresh, base):
    """Masque 1 = métal quand h>=z_thresh, 0 = air sinon."""
    return (h >= z_thresh-base).to(sim_dtype)

# ── 6) Fonction principale de simulation ─────────────────────────────────────
def simulate_spectrum(wl_list, angle_deg):
    """
    Pour chaque λ de wl_list, construit sim + perm_map
    dans l’ordre : air → sinus métallique → métal plein → SiO2.
    """
    R_list, T_list = [], []
    total_struct = 2*amplitude + 50.0  # hauteur totale des couches métalliques

    for wl in tqdm(wl_list, desc="λ"):
        freq = 1.0 / wl
        sim  = torcwa.rcwa(freq=freq, order=[4,4], L=L,
                           dtype=sim_dtype, device=device)

        # Carte permittivité (nz × ny)
        perm_map = torch.zeros((nz, ny), dtype=sim_dtype, device=device)

        # 6.1) Superstrat (air) : z >= total_struct
        mask_sup = (z >= total_struct)
        perm_map[mask_sup, :] = eps_air
        sim.add_input_layer(eps=eps_air)
        sim.add_output_layer(eps=eps_sub)

        # Angle d’incidence
        sim.set_incident_angle(inc_ang=angle_deg*np.pi/180, azi_ang=0.0)

        cumul = 0.0
        # 6.3) Couche métallique uniforme 50 nm
        z_bot, z_top = cumul, cumul + 50.0
        sim.add_layer(thickness=50.0, eps=eps_metal)
        idx = (z >= z_bot) & (z < z_top)
        perm_map[idx, :] = eps_metal
        cumul = z_top
        # 6.2) Couches sinusoïdales (de 0 à 2*amplitude)
        base = cumul
        for i in range(num_layers):
            z_bot = cumul
            z_top = cumul + dz
            z_mid = 0.5 * (z_bot + z_top)

            # 2D mask + permittivité de cette tranche
            mask2d    = create_pattern_layer(z_mid, base)  # [nx, ny]
            layer_eps = mask2d * eps_metal + (1-mask2d) * eps_air

            # ajout RCWA
            sim.add_layer(thickness=dz, eps=layer_eps)

            # remplissage perm_map pour z∈[z_bot,z_top[
            idx = (z >= z_bot) & (z < z_top)
            perm_map[idx, :] = layer_eps[x_idx, :]

            cumul = z_top


        # 6.4) Substrat (SiO₂) : z < 0
        mask_sub = (z < 0.0)
        perm_map[mask_sub, :] = eps_sub

        # 6.5) Affichage de la carte permittivité (Y vs Z)
        if False:
            plt.figure(figsize=(6,4))
            plt.imshow(perm_map.real.cpu().numpy(),
                    origin='lower',
                    extent=[0, Ly, -zmax, zmax],
                    aspect='auto')
            plt.colorbar(label='Re(ε)')
            plt.title(f"Permittivity map @ λ={wl:.1f}nm")
            plt.xlabel('y (nm)')
            plt.ylabel('z (nm)')
            plt.tight_layout()
            plt.pause(0.1)

        # 6.6) Calcul du S-matrix et R/T
        sim.solve_global_smatrix()
        R_pol, T_pol = [], []
        for pol in ['pp','ss','ps','sp','xx','xy','yx','yy']:
            R0 = sim.S_parameters(orders=[0,0], direction='forward',
                                port='reflection', polarization=pol, ref_order=[0,0])
            T0 = sim.S_parameters(orders=[0,0], direction='forward',
                                port='transmission', polarization=pol, ref_order=[0,0])
            R_pol.append((R0.abs().cpu()**2).item())
            T_pol.append((T0.abs().cpu()**2).item())

        R_list.append(R_pol)
        T_list.append(T_pol)

    return np.array(R_list), np.array(T_list), sim, perm_map

# ── 7) Exemple d’appel ────────────────────────────────────────────────────────
if __name__ == "__main__":
    wavelengths = torch.linspace(800.0, 2000.0, 100, dtype=geo_dtype)
    R_spec, T_spec, sim, perm_map = simulate_spectrum(wavelengths, angle_deg=30.0)
    # R et T sont de forme (len(wl_vec), 8)

    wavelengths = wavelengths.cpu()  # Move to CPU for plotting
    A_spec = 1.-R_spec - T_spec  # Absorption spectrum

    # Plot reflection and transmission vs wavelength
    if True:
        for idx, pol in enumerate(['xx', 'yy', 'pp', 'sp', 'ps', 'ss']):
            plt.figure(figsize=(6,4))
            plt.plot(wavelengths, T_spec[:, idx], label='Transmission (T)')
            plt.plot(wavelengths, R_spec[:, idx], label='Reflection (R)')
            #plt.plot(wavelengths, A_spec[:, idx], label='Absorption (A)')  
            plt.title(f"Reflectance and Transmittance at {pol} polarization")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Efficiency")
            plt.legend()
            plt.grid()
            plt.show()

    if False:
        # View XZ-plane fields and export
        sim.source_planewave(amplitude=[1.,0],direction='f')

        zmax = 2000
        z = torch.linspace(-1*zmax,zmax,1000,device=device)

        x_axis = torcwa.rcwa_geo.x.cpu()
        y_axis = torcwa.rcwa_geo.y.cpu()
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

