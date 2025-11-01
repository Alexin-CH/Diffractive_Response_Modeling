#!/bin/bash

set -e

working_dir=$(dirname "$0")
# echo "Working directory: $working_dir"

source "$working_dir/../../venv/bin/activate"

###############################

wavelength_start=200
wavelength_end=2500

angle_end=70
angle_start=10

amplitude_start=20
amplitude_end=100

period_start=100
period_end=10000

items=7

nh=30
discretization=256
perm_map=0

################################

# Calculate steps based on the number of items
wavelength_step=$(( (wavelength_end - wavelength_start) / (items - 1) ))
angle_step=$(( (angle_end - angle_start) / (items - 1) ))
amplitude_step=$(( (amplitude_end - amplitude_start) / (items - 1) ))
period_step=$(( (period_end - period_start) / (items - 1) ))

mkdir "$working_dir/samples" || echo "Directory 'samples' already exists." 

for wavelength in $(seq $wavelength_start $wavelength_step $wavelength_end); do
    for angle in $(seq $angle_start $angle_step $angle_end); do
        for amplitude in $(seq $amplitude_start $amplitude_step $amplitude_end); do
            for period in $(seq $period_start $period_step $period_end); do
                filename=\
"$working_dir/samples/sample\
.wl${wavelength}\
.ang${angle}\
.amp${amplitude}\
.per${period}\
.nh${nh}\
.dis${discretization}\
.json"
                if [ -f "$filename" ]; then
                    echo "File $filename already exists, skipping..."
                    continue
                fi
                python3 "$working_dir/process.py"  \
                    --wl $wavelength \
                    --ang $angle \
                    --sin_amplitude $amplitude \
                    --sin_period $period \
                    --nh $nh \
                    --discretization $discretization \
                    --filename "$filename"
                    # --perm_map $perm_map
            done
        done
    done
done

echo "done"
