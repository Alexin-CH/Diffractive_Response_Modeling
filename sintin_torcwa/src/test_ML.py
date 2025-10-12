import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torcwa

from struct_net import StructNet

#-----------------------------------------------------------------------------
#  Basic Setup
#-----------------------------------------------------------------------------

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Data types
sim_dtype = torch.complex64
geo_dtype = torch.float32

# Loss function
loss_fn = torch.nn.MSELoss()

#-----------------------------------------------------------------------------
#  Geometry Helper
#-----------------------------------------------------------------------------

def set_sinus_structure(amplitude, period, nxy=512, num_layers=50):
    """
    Build the RCWA geometry grid and height profile h(x,y)
    using amplitude and period (both torch scalars).
    Returns: L, ny, nz, z, eps_air, eps_sub, eps_metal,
             num_layers, dz, h, x_idx
    """
    # Unit‐cell dimensions (nm)
    Lx, Ly = period, period  # Square unit cell
    torcwa.rcwa_geo.dtype   = geo_dtype
    torcwa.rcwa_geo.device  = device
    torcwa.rcwa_geo.Lx      = Lx
    torcwa.rcwa_geo.Ly      = Ly
    torcwa.rcwa_geo.nx      = nxy
    torcwa.rcwa_geo.ny      = nxy
    torcwa.rcwa_geo.grid()

    # Materials
    eps_air   = torch.tensor(1.0, device=device)
    n_sub     = torch.tensor(1.44, device=device)
    eps_sub   = n_sub**2
    # TiN, as complex64
    eps_metal = torch.tensor(3.61 + 6.03j, dtype=sim_dtype, device=device)

    # Discretize sinusoidal corrugation
    dz = (2.0 * amplitude) / num_layers

    # z‐axis sample (for bookkeeping only)
    zmax = 1000.0
    z = torch.linspace(-zmax, zmax, steps=1000, device=device)
    nz = z.numel()
    ny = torcwa.rcwa_geo.ny

    # Precompute height profile h(x,y)
    X, Y = torch.meshgrid(torcwa.rcwa_geo.x,
                          torcwa.rcwa_geo.y,
                          indexing='xy')
    h = amplitude * torch.sin(2.0 * torch.pi * X / period) + amplitude

    # index for a central x‐slice (for debug/viewing)
    x_slice = Lx / 2.0
    x_idx = torch.argmin(torch.abs(torcwa.rcwa_geo.x - x_slice)).item()

    return (Lx, Ly), ny, nz, z, eps_air, eps_sub, eps_metal, \
           num_layers, dz, h, x_idx

#-----------------------------------------------------------------------------
#  Spectrum Simulator
#-----------------------------------------------------------------------------

def simulate_spectrum(wl_list, amplitude, period, inc_angle_deg=30.0, azi_angle_deg=0.0):
    """
    Compute R, T at each wavelength in wl_list.
    amplitude, period: torch scalars with requires_grad=True.
    Returns: R, T tensor shape = (len(wl_list),)
    """
    # Build the fixed geometry once
    L, ny, nz, z, eps_air, eps_sub, eps_metal, \
    num_layers, dz, h, x_idx = set_sinus_structure(amplitude, period)

    R_list, T_list = [], []

    tqdm_wl = tqdm(wl_list, desc='Simulating Spectrum', leave=False)
    for wl in tqdm_wl:
        freq = 1.0 / wl

        # Initialize RCWA simulation
        sim = torcwa.rcwa(freq=freq,
                          order=[0, 3],
                          L=L,
                          dtype=sim_dtype,
                          device=device)

        # Superstrate + substrate
        sim.add_input_layer(eps=eps_air)
        sim.add_output_layer(eps=eps_sub)

        # Incident angle
        inc_ang = inc_angle_deg * torch.pi / 180.0
        azi_ang = azi_angle_deg * torch.pi / 180.0
        sim.set_incident_angle(inc_ang=inc_ang, azi_ang=azi_ang)

        # A base metal film (e.g. 50 nm)
        sim.add_layer(thickness=50.0, eps=eps_metal)

        # Add the discretized sinusoid
        for i in tqdm_wl:
            tqdm_wl.set_description(f'Adding layer {i+1}/{num_layers} for λ={wl:.1f} nm')
            z_mid = dz * (i + 0.5)

            # binary mask at this slice
            mask = (h >= z_mid).to(sim_dtype)

            # local permittivity: metal inside mask, air outside
            eps_local = mask * eps_metal + (1.0 - mask) * eps_air

            sim.add_layer(thickness=dz, eps=eps_local)

        # Finally solve
        sim.solve_global_smatrix()

        # Extract reflection (order 0,0)
        pol = 'xx'
        R0 = sim.S_parameters(orders=[0,0],
                               direction='forward',
                               port='r',
                               polarization=pol,
                               ref_order=[0,0])

        T0 = sim.S_parameters(orders=[0,0],
                               direction='forward',
                               port='t',
                               polarization=pol,
                               ref_order=[0,0])

        # store intensity |R0|^2
        R_list.append((R0.abs()**2).squeeze())
        T_list.append((T0.abs()**2).squeeze())

    # stack preserves gradient links
    R = torch.stack(R_list)   # shape (n_wl,)
    T = torch.stack(T_list)   # shape (n_wl,)
    return R, T

#-----------------------------------------------------------------------------
#  Main Training + Plot
#-----------------------------------------------------------------------------

if __name__ == '__main__':
    # Define wavelengths
    n = 30
    wavelengths = torch.linspace(1000.0, 2000.0, n,
                                 dtype=geo_dtype, device=device)

    # Learnable parameter
    # sim_params = torch.nn.ParameterDict({
    #     'inc_angle_deg': torch.tensor(30.0, dtype=geo_dtype, device=device, requires_grad=True),
    #     'period': torch.tensor(1000.0, dtype=geo_dtype, device=device, requires_grad=True),
    #     'amplitude': torch.tensor(50.0, dtype=geo_dtype, device=device, requires_grad=True)
    # })

    model = StructNet(input_dim1=n, input_dim2=n, output_dim=3).to(device)

    # Define boundaries for the parameters
    ang_bounds = torch.tensor((0.0, 60.0), dtype=geo_dtype, device=device)  # degrees   
    per_bounds = torch.tensor((100.0, 2000.0), dtype=geo_dtype, device=device)  # nm
    amp_bounds = torch.tensor((20.0, 100.0), dtype=geo_dtype, device=device)  # nm

    # # First compute the initial spectrum
    # print("Initial parameters:")
    # print(f"  Amplitude = {amplitude.item():.2f} nm")
    # print(f"  Period    = {period.item():.2f} nm")
    # Rpp_initial = simulate_spectrum(wavelengths, amplitude, period)

    # # Plot initial derivative spectrum (derivative give the slope) and the spectrum in a subplot
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(wavelengths.cpu(), Rpp_initial.cpu().detach(), 'b-o')  
    # plt.title('Initial R_pp Spectrum')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('R_pp intensity')
    # plt.grid(True)
    # plt.subplot(1, 2, 2)
    # plt.plot(wavelengths.cpu()[1:], Rpp_initial.cpu().detach().diff(), 'r-o')
    # plt.title('Initial R_pp Derivative Spectrum')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('dR_pp/dλ')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # exit(0)  # Exit early for debugging

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.66,
        patience=50,
        min_lr=1e-6
    )

    # Create bandpass for the wavelengths
    passband = [1400, 1600]  # nm
    R0 = torch.zeros_like(wavelengths, dtype=geo_dtype, device=device)
    R0[(wavelengths < passband[0]) | (wavelengths > passband[1])] = 1
    T0 = 1 - R0  # Complementary bandpass

    # # Plot the bandpass filter
    # plt.figure(figsize=(6, 4))
    # plt.plot(wavelengths.cpu(), R_bp.cpu(), 'g--', label='Bandpass Filter')
    # plt.plot(wavelengths.cpu(), T_cb.cpu(), 'r--', label='Complementary Bandpass')
    # plt.title('Bandpass Filter for Wavelengths')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Filter Intensity')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # exit(0)  # Exit early for debugging

    # Training loop
    epochs = 1000
    tqdm_epoch = tqdm(range(epochs), desc='Training RCWA Spectrum')
    for epoch in tqdm_epoch:
        optimizer.zero_grad()

        # Get learnable parameters
        sim_params = model(R0, T0)
        sim_params = {
            'inc_angle_deg': sim_params[0].to(geo_dtype).to(device),
            'period': sim_params[1].to(geo_dtype).to(device),
            'amplitude': sim_params[2].to(geo_dtype).to(device)
        }
        # Forward: get Rpp for all λ
        R, T = simulate_spectrum(wavelengths,
                                 sim_params['amplitude'],
                                 sim_params['period'],
                                 inc_angle_deg=sim_params['inc_angle_deg'])

        # Loss: minimize the smallest reflectivity
        bc_amp, bc_per, bc_ang = False, False, False
        if sim_params['amplitude'] < amp_bounds[0] or sim_params['amplitude'] > amp_bounds[1]:
            bc_amp = True
        if sim_params['period'] < per_bounds[0] or sim_params['period'] > per_bounds[1]:
            bc_per = True
        if sim_params['inc_angle_deg'] < ang_bounds[0] or sim_params['inc_angle_deg'] > ang_bounds[1]:
            bc_ang = True

        # If boundary condition is violated, add a penalty
        if bc_amp:
            loss_bc_amp = torch.abs(sim_params['amplitude'] - amp_bounds[0]) + \
                            torch.abs(sim_params['amplitude'] - amp_bounds[1])
        else:
            loss_bc_amp = torch.tensor(0.0, dtype=geo_dtype, device=device)
        if bc_per:
            loss_bc_per = torch.abs(sim_params['period'] - per_bounds[0]) + \
                            torch.abs(sim_params['period'] - per_bounds[1])
        else:
            loss_bc_per = torch.tensor(0.0, dtype=geo_dtype, device=device)
        if bc_ang:
            loss_bc_ang = torch.abs(sim_params['inc_angle_deg'] - ang_bounds[0]) + \
                            torch.abs(sim_params['inc_angle_deg'] - ang_bounds[1])
        else:
            loss_bc_ang = torch.tensor(0.0, dtype=geo_dtype, device=device)
        
        # Boundary condition loss
        loss_bc = loss_bc_amp + loss_bc_per + loss_bc_ang

        # Compute the total loss
        loss = loss_fn(R, R0) + loss_fn(T, T0) + loss_bc + \
            loss_fn(R.diff(), R0.diff()) + \
            loss_fn(T.diff(), T0.diff())
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        lr = scheduler.get_last_lr()[0]
        tqdm_epoch.set_description(f'Training RCWA Spectrum, Loss: {loss.item():.4f}, Amp: {sim_params["amplitude"].item():.2f} nm, Per: {sim_params["period"].item():.2f} nm, Inc: {sim_params["inc_angle_deg"].item():.2f} deg, LR: {lr:.6f}')

    # Final optimized parameters
    amp_opt = sim_params['amplitude'].item()
    per_opt = sim_params['period'].item()
    ang_opt = sim_params['inc_angle_deg'].item()
    print("Optimization complete!")
    print(f"  Amplitude = {amp_opt:.2f} nm")
    print(f"  Period    = {per_opt:.2f} nm")

    # Recompute spectrum for plotting
    wavelengths = torch.linspace(1000.0, 2000.0, 100, dtype=geo_dtype, device=device)
    wavelengths_cpu = wavelengths.cpu()
    R_final, T_final = simulate_spectrum(wavelengths, amp_opt, per_opt, inc_angle_deg=ang_opt).cpu().detach()

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(wavelengths_cpu, R_final, 'b-o', label='R_pp')
    plt.plot(wavelengths_cpu, T_final, 'r-o', label='T_pp')
    plt.title('Optimized R_pp Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('R_pp intensity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
