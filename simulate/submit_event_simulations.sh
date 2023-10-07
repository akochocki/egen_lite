#!/bin/bash

module purge > /dev/null 2>&1
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
module load CUDA/11.4.2
/mnt/home/kochocki/egen_lite/icetray_build/env-shell.sh

######### Process NuE CC, 25 events each #########
sbatch --array=1-250 -A deyoungbuyin --gres=gpu:1 --mem=26000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/sim_NuE_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/sim_NuE_%a.out /mnt/home/kochocki/egen_lite/simulate/simulate_events.py -g '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz'

######### Process MuonGun, 5 events each #########
sbatch --array=1-250 -A deyoungbuyin --gres=gpu:1 --mem=24000 --time=60:00:00 --error=/mnt/home/kochocki/egen_lite/logs/sim_MuonGun_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/sim_MuonGun_%a.out /mnt/home/kochocki/egen_lite/simulate/simulate_events.py -g '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz' --MuonGun=True

