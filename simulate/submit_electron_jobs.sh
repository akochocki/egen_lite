#!/bin/bash

module purge > /dev/null 2>&1
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
module load CUDA/11.4.2
/mnt/home/kochocki/egen_lite/icetray_build/env-shell.sh


# Aim to submit 40k jobs on scavenger
sbatch --array=38001-39000  -A deyoungbuyin --qos=scavenger --gres=gpu:1 --mem=15000 --time=8:00:00 --error=/mnt/home/kochocki/egen_lite/logs/gen_10TeV_%a.err --output=/mnt/home/kochocki/egen_lite/logs/gen_10TeV_%a.out /mnt/home/kochocki/egen_lite/simulate/sim_electrons.py -o Gen1k_10TeV_EMinus_FTP_V2_
