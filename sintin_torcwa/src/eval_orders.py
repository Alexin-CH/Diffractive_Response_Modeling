# Python script: RCWA simulation of a layered metasurface using TORCWA
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from rcwa import setup, RCWAArgs

def eval_orders(order_list, device='cpu'):
    """Compute reflection and transmission spectra over wavelengths at a given angle."""
    R_list = []
    T_list = []
    time_list = []
    tqdm_order = tqdm(order_list)
    for order in tqdm_order:
        t0 = time.time()
        tqdm_order.set_description(f"Simulating orders={order}")
        args = RCWAArgs(
            wl=1500.0,
            ang=30.0,
            nh=order,
            discretization=256,
            sin_amplitude=55.0,
            sin_period=1000.0
        )
        sim, _  = setup(args=args, device = device)
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


# Main evaluation loop
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
n_avg = 5  # Number of averages for timing

print(f"Available devices: {devices}")
print(f"Number of averages for timing: {n_avg}")

times = {}
for device in devices:
    print(f"\nEvaluating on device: {device}")
    orders = np.arange(0, 40)

    # Repeat each measurement 3 times and average
    times[device] = []
    for _ in range(n_avg):
        R_spec, T_spec, exec_time = eval_orders(orders, device=device)
        times[device].append(exec_time)
    
    times[device] = np.mean(times[device], axis=0)

# Plot computation times[device] for each order
plt.figure(figsize=(6,4))
for device in devices:
    plt.plot(orders, times[device], marker='o', label=device)
plt.title("Computation Time vs Order")
plt.xlabel("Order")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid()
plt.show()

# Plot reflection and transmission vs wavelength
for idx, pol in enumerate(['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']):
    plt.figure(figsize=(6,4))
    plt.plot(orders, R_spec[:, idx], label='Reflection (R)')
    plt.plot(orders, T_spec[:, idx], label='Transmission (T)')
    plt.title(f"Reflectance and Transmittance at {pol} polarization")
    plt.xlabel("Order")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid()
    # plt.savefig(f"{pol}pol_vs_orders.png", dpi=300)
    plt.show()
