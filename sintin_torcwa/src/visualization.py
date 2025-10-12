import torch
import numpy as np
import matplotlib.pyplot as plt

def S_parameters(sim,orders,*,direction='forward',port='transmission',polarization='xx',ref_order=[0,0],power_norm=True,evanscent=1e-3):
        '''
            Return S-parameters.

            Parameters
            - orders: selected orders (Recommended shape: Nx2)

            - direction: set the direction of light propagation ('f', 'forward' / 'b', 'backward')
            - port: set the direction of light propagation ('t', 'transmission' / 'r', 'reflection')
            - polarization: set the input and output polarization of light ((output,input) xy-pol: 'xx' / 'yx' / 'xy' / 'yy' , ps-pol: 'pp' / 'sp' / 'ps' / 'ss' )
            - ref_order: set the reference for calculating S-parameters (Recommended shape: Nx2)
            - power_norm: if set as True, the absolute square of S-parameters are corresponds to the ratio of power
            - evanescent: Criteria for judging the evanescent field. If power_norm=True and real(kz_norm)/imag(kz_norm) < evanscent, function returns 0 (default = 1e-3)

            Return
            - S-parameters (torch.Tensor)
        '''

        orders = torch.as_tensor(orders,dtype=torch.int64,device=sim._device).reshape([-1,2])

        if direction in ['f', 'forward']:
            direction = 'forward'
        elif direction in ['b', 'backward']:
            direction = 'backward'
        else:
            warnings.warn('Invalid propagation direction. Set as forward.',UserWarning)
            direction = 'forward'

        if port in ['t', 'transmission']:
            port = 'transmission'
        elif port in ['r', 'reflection']:
            port = 'reflection'
        else:
            warnings.warn('Invalid port. Set as tramsmission.',UserWarning)
            port = 'transmission'

        if polarization not in ['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']:
            warnings.warn('Invalid polarization. Set as xx.',UserWarning)
            polarization = 'xx'

        ref_order = torch.as_tensor(ref_order,dtype=torch.int64,device=sim._device).reshape([1,2])

        # Matching order indices
        order_indices = sim._matching_indices(orders)
        ref_order_index = sim._matching_indices(ref_order)

        if polarization in ['xx', 'yx', 'xy', 'yy']:
            # Matching order indices with polarization
            if polarization == 'yx' or polarization == 'yy':
                order_indices = order_indices + sim.order_N
            if polarization == 'xy' or polarization == 'yy':
                ref_order_index = ref_order_index + sim.order_N

            # power normalization factor
            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(sim.eps_in*sim.mu_in - sim.Kx_norm_dn**2 - sim.Ky_norm_dn**2)
                is_evanescent_in = torch.abs(torch.real(Kz_norm_dn_in_complex) / torch.imag(Kz_norm_dn_in_complex)) < evanscent
                Kz_norm_dn_in = torch.where(is_evanescent_in,torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),torch.real(Kz_norm_dn_in_complex))
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in,Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(sim.eps_out*sim.mu_out - sim.Kx_norm_dn**2 - sim.Ky_norm_dn**2)
                is_evanescent_out = torch.abs(torch.real(Kz_norm_dn_out_complex) / torch.imag(Kz_norm_dn_out_complex)) < evanscent
                Kz_norm_dn_out = torch.where(is_evanescent_out,torch.real(torch.zeros_like(Kz_norm_dn_out_complex)),torch.real(Kz_norm_dn_out_complex))
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out,Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack((torch.real(sim.Kx_norm_dn),torch.real(sim.Kx_norm_dn)))
                Ky_norm_dn = torch.hstack((torch.real(sim.Ky_norm_dn),torch.real(sim.Ky_norm_dn)))

                if polarization == 'xx':
                    numerator_pol, denominator_pol = Kx_norm_dn, Kx_norm_dn
                elif polarization == 'xy':
                    numerator_pol, denominator_pol = Kx_norm_dn, Ky_norm_dn
                elif polarization == 'yx':
                    numerator_pol, denominator_pol = Ky_norm_dn, Kx_norm_dn
                elif polarization == 'yy':
                    numerator_pol, denominator_pol = Ky_norm_dn, Ky_norm_dn

                if direction == 'forward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'forward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'backward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == 'backward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                normalization = torch.sqrt((1+(numerator_pol[order_indices]/numerator_kz[order_indices])**2)/(1+(denominator_pol[ref_order_index]/denominator_kz[ref_order_index])**2))
                normalization = normalization * torch.sqrt(numerator_kz[order_indices]/denominator_kz[ref_order_index])
            else:
                normalization = 1.

            # Get S-parameters
            if direction == 'forward' and port == 'transmission':
                S = sim.S[0][order_indices,ref_order_index] * normalization
            elif direction == 'forward' and port == 'reflection':
                S = sim.S[1][order_indices,ref_order_index] * normalization
            elif direction == 'backward' and port == 'reflection':
                S = sim.S[2][order_indices,ref_order_index] * normalization
            elif direction == 'backward' and port == 'transmission':
                S = sim.S[3][order_indices,ref_order_index] * normalization

            S = torch.where(torch.isinf(S),torch.zeros_like(S),S)
            S = torch.where(torch.isnan(S),torch.zeros_like(S),S)

            return S
        
        elif polarization in ['pp', 'sp', 'ps', 'ss']:
            if direction == 'forward' and port == 'transmission':
                idx = 0
                order_sign, ref_sign = 1, 1
                order_k0_norm2 = sim.eps_out * sim.mu_out
                ref_k0_norm2 = sim.eps_in * sim.mu_in
            elif direction == 'forward' and port == 'reflection':
                idx = 1
                order_sign, ref_sign = -1, 1
                order_k0_norm2 = sim.eps_in * sim.mu_in
                ref_k0_norm2 = sim.eps_in * sim.mu_in
            elif direction == 'backward' and port == 'reflection':
                idx = 2
                order_sign, ref_sign = 1, -1
                order_k0_norm2 = sim.eps_out * sim.mu_out
                ref_k0_norm2 = sim.eps_out * sim.mu_out
            elif direction == 'backward' and port == 'transmission':
                idx = 3
                order_sign, ref_sign = -1, -1
                order_k0_norm2 = sim.eps_in * sim.mu_in
                ref_k0_norm2 = sim.eps_out * sim.mu_out

            order_Kx_norm_dn = sim.Kx_norm_dn[order_indices]
            order_Ky_norm_dn = sim.Ky_norm_dn[order_indices]
            order_Kt_norm_dn = torch.sqrt(order_Kx_norm_dn**2 + order_Ky_norm_dn**2)
            order_Kz_norm_dn = order_sign*torch.abs(torch.real(torch.sqrt(order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2)))
            order_Kz_norm_dn_complex = torch.sqrt(order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2)
            order_is_evanescent = torch.abs(torch.real(order_Kz_norm_dn_complex) / torch.imag(order_Kz_norm_dn_complex)) < evanscent

            order_inc_angle = torch.atan2(torch.real(order_Kt_norm_dn),order_Kz_norm_dn)
            order_azi_angle = torch.atan2(torch.real(order_Ky_norm_dn),torch.real(order_Kx_norm_dn))

            ref_Kx_norm_dn = sim.Kx_norm_dn[ref_order_index]
            ref_Ky_norm_dn = sim.Ky_norm_dn[ref_order_index]
            ref_Kt_norm_dn = torch.sqrt(ref_Kx_norm_dn**2 + ref_Ky_norm_dn**2)
            ref_Kz_norm_dn = ref_sign*torch.abs(torch.real(torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)))
            ref_Kz_norm_dn_complex = torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)
            ref_is_evanescent = torch.abs(torch.real(ref_Kz_norm_dn_complex) / torch.imag(ref_Kz_norm_dn_complex)) < evanscent

            ref_inc_angle = torch.atan2(torch.real(ref_Kt_norm_dn),ref_Kz_norm_dn)
            ref_azi_angle = torch.atan2(torch.real(ref_Ky_norm_dn),torch.real(ref_Kx_norm_dn))

            xx = sim.S[idx][order_indices,ref_order_index]
            xy = sim.S[idx][order_indices,ref_order_index+sim.order_N]
            yx = sim.S[idx][order_indices+sim.order_N,ref_order_index]
            yy = sim.S[idx][order_indices+sim.order_N,ref_order_index+sim.order_N]

            xx = torch.where(order_is_evanescent,torch.zeros_like(xx),xx)
            xy = torch.where(order_is_evanescent,torch.zeros_like(xy),xy)
            yx = torch.where(order_is_evanescent,torch.zeros_like(yx),yx)
            yy = torch.where(order_is_evanescent,torch.zeros_like(yy),yy)

            if ref_is_evanescent:
                S = torch.zeros_like(xx)
                return S

            if polarization == 'pp':
                S = torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * xx +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * yx +\
                    torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * xy +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * yy
            elif polarization == 'ps':
                S = torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * (-1)*torch.sin(ref_azi_angle) * xx +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * (-1)*torch.sin(ref_azi_angle) * yx +\
                    torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_azi_angle) * xy +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_azi_angle) * yy
            elif polarization == 'sp':
                S = -torch.sin(order_azi_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * xx +\
                    torch.cos(order_azi_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * yx +\
                    -torch.sin(order_azi_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * xy +\
                    torch.cos(order_azi_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * yy
            elif polarization == 'ss':
                S = -torch.sin(order_azi_angle) * (-1)*torch.sin(ref_azi_angle) * xx +\
                    torch.cos(order_azi_angle) * (-1)*torch.sin(ref_azi_angle) * yx +\
                    -torch.sin(order_azi_angle) * torch.cos(ref_azi_angle) * xy +\
                    torch.cos(order_azi_angle) * torch.cos(ref_azi_angle) * yy

            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(sim.eps_in*sim.mu_in - sim.Kx_norm_dn**2 - sim.Ky_norm_dn**2)
                is_evanescent_in = torch.abs(torch.real(Kz_norm_dn_in_complex) / torch.imag(Kz_norm_dn_in_complex)) < evanscent
                Kz_norm_dn_in = torch.where(is_evanescent_in,torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),torch.real(Kz_norm_dn_in_complex))
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in,Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(sim.eps_out*sim.mu_out - sim.Kx_norm_dn**2 - sim.Ky_norm_dn**2)
                is_evanescent_out = torch.abs(torch.real(Kz_norm_dn_out_complex) / torch.imag(Kz_norm_dn_out_complex)) < evanscent
                Kz_norm_dn_out = torch.where(is_evanescent_out,torch.abs(torch.real(Kz_norm_dn_out_complex)),torch.real(Kz_norm_dn_out_complex))
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out,Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack((torch.real(sim.Kx_norm_dn),torch.real(sim.Kx_norm_dn)))
                Ky_norm_dn = torch.hstack((torch.real(sim.Ky_norm_dn),torch.real(sim.Ky_norm_dn)))

                if direction == 'forward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'forward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'backward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == 'backward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                normalization = torch.sqrt(numerator_kz[order_indices]/denominator_kz[ref_order_index])
            else:
                normalization = 1.

            S = torch.where(torch.isinf(S),torch.zeros_like(S),S)
            S = torch.where(torch.isnan(S),torch.zeros_like(S),S)

            return S * normalization
        
        else:
            return None


def main():
    dataset = torch.load('dataset_sim_SinTiN_TORCWA_50_400_2000_50_0_70_3_256.pt')
    pol = 'ps'  # Change this to 'xx', 'xy', 'yx', 'yy', 'pp', 'sp', 'ps', or 'ss' as needed
    WL, ANG, R = [], [], []
    for i in range(len(dataset)):
        sim = dataset[i]['sim']
        R0 = sim.S_parameters(orders=[0,0],
                               direction='forward',
                               port='reflection',
                               polarization=pol,
                               ref_order=[0,0])
        
        R.append(R0.abs().cpu()**2)
        WL.append(dataset[i]['wl'].cpu())
        ANG.append(dataset[i]['ang'].cpu())
    
    # Plotting the results as a 2D map
    R = torch.stack(R).squeeze()
    WL = torch.stack(WL).squeeze()
    ANG = torch.stack(ANG).squeeze()
    plt.figure(figsize=(10, 6))
    plt.imshow(R, aspect='auto', extent=[ANG.min(), ANG.max(), WL.min(), WL.max()], origin='lower', cmap='viridis')
    plt.colorbar(label='Reflectance')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Reflectance Map')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
