#!/bin/bash

set -e

working_dir=$(dirname "$0")
# echo "Working directory: $working_dir"

source "$working_dir/../../venv/bin/activate"

## Define ranges ###############
wavelength_start=200
wavelength_end=2500

angle_end=70
angle_start=10

amplitude_start=20
amplitude_end=100

period_start=100
period_end=10000

items=10

## Define simulation params ####
nh=30
discretization=256
################################

# Calculate steps based on the number of items
wavelength_step=$(( (wavelength_end - wavelength_start) / (items - 1) ))
angle_step=$(( (angle_end - angle_start) / (items - 1) ))
amplitude_step=$(( (amplitude_end - amplitude_start) / (items - 1) ))
period_step=$(( (period_end - period_start) / (items - 1) ))

mkdir "$working_dir/data-outputs" || echo "Directory 'data-outputs' already exists." 

# Loop through angles and wavelengths
for wavelength in $(seq $wavelength_start $wavelength_step $wavelength_end); do
    for angle in $(seq $angle_start $angle_step $angle_end); do
        for amplitude in $(seq $amplitude_start $amplitude_step $amplitude_end); do
            for period in $(seq $period_start $period_step $period_end); do
                filename=\
"$working_dir/data-outputs/sample\
.wl${wavelength}\
.ang${angle}\
.amp${amplitude}\
.per${period}\
.nh${nh}\
.dis${discretization}\
.pt"
                if [ -f "$filename" ]; then
                    echo "File $filename already exists, skipping..."
                    continue
                fi
                #amplitude=55
                python3 "$working_dir/process.py"  \
                    --wl $wavelength \
                    --ang $angle \
                    --sin_amplitude $amplitude \
                    --sin_period $period \
                    --nh $nh \
                    --discretization $discretization \
                    --filename "$filename"
            done
        done
    done
done

echo "done"
