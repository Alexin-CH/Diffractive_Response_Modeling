#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process.py — Generate one RCWA sample for given (wavelength, angle, nh, discretization)
Usage:
    python3 process.py --wl 500 --ang 30 --nh 100 --discretization 64
"""
import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm
import torcwa



def main():
    # 1) Parse command‐line arguments
    parser = argparse.ArgumentParser(
        description="Generate and save one RCWA sample for a single wavelength and angle"
    )
    parser.add_argument("--wl", type=float, required=True,
                        help="Wavelength in nm")
    parser.add_argument("--ang", type=float, required=True,
                        help="Incidence angle in degrees")
    parser.add_argument("--nh", type=int, default=50,
                        help="Number of Fourier harmonics")
    parser.add_argument("--discretization", type=int, default=128,
                        help="Grid size in x and y")
    parser.add_argument("--sin_amplitude", type=float, default=55.0,
                        help="Amplitude of sinusoidal corrugation in nm")
    parser.add_argument("--sin_period", type=float, default=1000.0,
                        help="Period of sinusoidal corrugation in nm")
    parser.add_argument("--filename", type=str, default="data_sim.pt",
                        help="Output filename (default: data_sim.pt)")
    
    args = parser.parse_args()

    t0 = time.time()

    # 2) Device and dtypes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    sim_dtype = torch.complex64
    geo_dtype = torch.float32

    # 3) Unit cell size (nm)
    Lx, Ly = 1000.0, 1000.0
    L = [Lx, Ly]

    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = Lx
    torcwa.rcwa_geo.Ly = Ly
    torcwa.rcwa_geo.nx = args.discretization
    torcwa.rcwa_geo.ny = args.discretization
    torcwa.rcwa_geo.grid()

    # 4) Materials
    eps_air   = torch.tensor(1.0, dtype=sim_dtype, device=device)
    n_sub     = 1.44
    eps_sub   = torch.tensor(n_sub**2, dtype=sim_dtype, device=device)
    eps_metal = torch.tensor(3.61 + 6.03j, dtype=sim_dtype, device=device)

    # 5) Sinusoidal corrugation parameters
    amplitude  = args.sin_amplitude  # nm
    period     = args.sin_period    # nm
    num_layers = 30
    dz         = (2 * amplitude) / num_layers

    # 6) z‐axis and central x‐slice index
    zmax = 1000.0  # nm
    z = torch.linspace(-zmax, zmax, 500, device=device)
    nz = z.numel()
    x_slice = Lx / 2
    x_idx = torch.argmin(torch.abs(torcwa.rcwa_geo.x - x_slice)).item()

    # Precompute height profile h(x,y)
    X, Y = torch.meshgrid(torcwa.rcwa_geo.x,
                          torcwa.rcwa_geo.y,
                          indexing="xy")
    h = amplitude * torch.sin(2 * np.pi * X / period) + amplitude

    def create_pattern_layer(z_mid, base):
        # Returns a mask = 1 where metal, 0 where air
        return (h >= (z_mid - base)).to(sim_dtype)

    # 7) Build and run RCWA simulation
    freq = 1.0 / args.wl  # in 1/nm
    sim = torcwa.rcwa(freq=freq,
                      order=[args.nh, args.nh],
                      L=L,
                      dtype=sim_dtype,
                      device=device)

    # Prepare permittivity map (z vs y) for diagnostics
    perm_map = torch.zeros((nz, torcwa.rcwa_geo.ny),
                           dtype=sim_dtype,
                           device=device)

    # 7.1) Superstrate (air)
    total_struct = 2 * amplitude + 50.0
    mask_sup = (z >= total_struct)
    perm_map[mask_sup, :] = eps_air
    sim.add_input_layer(eps=eps_air)
    sim.add_output_layer(eps=eps_sub)

    # 7.2) Incident wave and angle
    sim.set_incident_angle(inc_ang=args.ang * np.pi/180, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")

    # 7.3) Uniform metal base layer (50 nm)
    cumul = 0.0
    z_bot, z_top = cumul, cumul + 50.0
    sim.add_layer(thickness=50.0, eps=eps_metal)
    idx = (z >= z_bot) & (z < z_top)
    perm_map[idx, :] = eps_metal
    cumul = z_top

    # 7.4) Sinusoidal layers
    base = cumul
    for _ in range(num_layers):
        z_bot = cumul
        z_top = cumul + dz
        z_mid = 0.5 * (z_bot + z_top)

        mask2d = create_pattern_layer(z_mid, base)
        layer_eps = mask2d * eps_metal + (1 - mask2d) * eps_air

        sim.add_layer(thickness=dz, eps=layer_eps)

        idx = (z >= z_bot) & (z < z_top)
        perm_map[idx, :] = layer_eps[x_idx, :]
        cumul = z_top

    # 7.5) Substrate (SiO₂)
    mask_sub = (z < 0.0)
    perm_map[mask_sub, :] = eps_sub

    # 7.6) Solve S‐matrix and compute reflectance/transmittance
    sim.solve_global_smatrix()
    # R_pol, T_pol = {}, {}
    # for pol in ['pp','ss','ps','sp','xx','xy','yx','yy']:
    #     R0 = sim.S_parameters(orders=[0,0],
    #                           direction='forward',
    #                           port='reflection',
    #                           polarization=pol,
    #                           ref_order=[0,0])
    #     T0 = sim.S_parameters(orders=[0,0],
    #                           direction='forward',
    #                           port='transmission',
    #                           polarization=pol,
    #                           ref_order=[0,0])
    #     R_pol[pol] = (R0.abs().cpu()**2).item()
    #     T_pol[pol] = (T0.abs().cpu()**2).item()

    # # 7.7) Extract E and H fields in the Y–Z plane
    # [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_yz(torcwa.rcwa_geo.y, z, L[1]/2)

    # fields = {
    #     'Ex': Ex.cpu(),
    #     'Ey': Ey.cpu(),
    #     'Ez': Ez.cpu(),
    #     'Hx': Hx.cpu(),
    #     'Hy': Hy.cpu(),
    #     'Hz': Hz.cpu(),
    # }

    # # 8) Package all data, including the permittivity map
    # sample = {
    #     'theta':         args.ang,          # incidence angle
    #     'wavelength':    args.wl,
    #     'reflectance':   R_pol,
    #     'transmittance': T_pol,
    #     'fields':        fields,
    #     'perm_map':      perm_map.cpu(),    # permittivity map (z vs y)
    # }

    sample = sim.export_data_dict()

    # 9) Save to disk
    out_dir = "data-outputs"
    os.makedirs(out_dir, exist_ok=True)
    filename = args.filename
    out_path = os.path.join(filename)
    torch.save(sample, out_path)
    print(f"Sample saved to: {out_path} ({time.time()-t0:.2f}s)")

    perm_map_title = f"data_sim.perm_map.{int(amplitude)}_{int(period)}_{int(zmax)}.pt"
    perm_map_path = os.path.join(out_dir, perm_map_title)
    if not os.path.exists(perm_map_path):
        torch.save(perm_map.cpu(), perm_map_path)
        print(f"Permittivity map saved to: {perm_map_path}")

if __name__ == "__main__":
    main()
