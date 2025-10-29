#!/bin/sh

set -e

. ../../../venv/bin/activate

# Define ranges
angle_start=10
angle_end=70

wavelength_start=200
wavelength_end=2500

amplitude_start=20
amplitude_end=100

period_start=100
period_end=10000

 # # # # # # # # # # # # # #
# # # # # # # # # # # # # #
items=11
nh=20
discretization=256
 # # # # # # # # # # # # #
# # # # # # # # # # # # #

# Calculate steps based on the number of items
amplitude_step=$(( (wavelength_end - wavelength_start) / (items - 1) ))
period_step=$(( (period_end - period_start) / (items - 1) ))
angle_step=$(( (angle_end - angle_start) / (items - 1) ))
wavelength_step=$(( (wavelength_end - wavelength_start) / (items - 1) ))

mkdir "data-outputs" || echo "Directory 'data-outputs' already exists." 

# Loop through angles and wavelengths
for amplitude in $(seq $amplitude_start $amplitude_step $amplitude_end); do
    for period in $(seq $period_start $period_step $period_end); do
        for angle in $(seq $angle_start $angle_step $angle_end); do
            for wavelength in $(seq $wavelength_start $wavelength_step $wavelength_end); do
                # filename = f"data_sim.{int(args.wl)}_{int(args.ang)}.{args.nh}_{args.discretization}.{int(amplitude)}_{int(period)}_{int(zmax)}.pt"
                filename="data-outputs/data_sim.${wavelength}.${angle}.${nh}.${discretization}.${amplitude}.${period}.pt"
                if [ -f "$filename" ]; then
                    echo "File $filename already exists, skipping..."
                    continue
                fi
                #amplitude=55
                python3 process.py --wl $wavelength --ang $angle --nh $nh --discretization $discretization --sin_amplitude $amplitude --sin_period $period --filename "$filename"
            done
        done
    done
done

echo "done"
