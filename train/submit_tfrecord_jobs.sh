#!/bin/bash

module purge
module load CUDA/11.4.2

PYTHONPATH=/mnt/home/kochocki/prefix/bin:/mnt/home/kochocki/prefix/lib/python3.7/site-packages:/mnt/home/kochocki/prefix/lib/python3.6/site-packages:/mnt/home/kochocki/nuSQuIDS_build/resources/python/bindings:/mnt/home/kochocki/prefix:/mnt/home/kochocki/prefix/lib64/python3.6/site-packages

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`

PYTHONPATH=/mnt/home/kochocki/prefix/lib64/:$PYTHONPATH
LD_LIBRARY_PATH=/mnt/home/kochocki/prefix/lib64/:$LD_LIBRARY_PATH

# Aim to submit 8k jobs on scavenger
# --gres=gpu:1
sbatch --array=7101-8000 -A deyoungbuyin --gres=gpu:1 --qos=scavenger --mem=30000 --time=4:00:00 --error=/mnt/home/kochocki/egen_lite/logs/record_10TeV_%a.err --output=/mnt/home/kochocki/egen_lite/logs/record_10TeV_%a.out /mnt/home/kochocki/egen_lite/train/create_tf_records.py
