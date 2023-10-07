#!/bin/bash

module purge > /dev/null 2>&1
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
module load CUDA/11.4.2
/mnt/home/kochocki/egen_lite/icetray_build/env-shell.sh

sbatch --array=1 -A deyoungbuyin --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/plot_cnn_comp_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/plot_cnn_comp_%a.out /mnt/home/kochocki/egen_lite/plot/plot_cnn_comparisons.py

sbatch --array=1 -A deyoungbuyin --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/plot_cnn_spline_comp_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/plot_cnn_spline_comp_%a.out /mnt/home/kochocki/egen_lite/plot/plot_cnn_spline_comparisons.py 'ftp_v2_1_3_25_500'

sbatch --array=1-10 -A deyoungbuyin --gres=gpu:1  --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/get_time_stats_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/get_time_stats_%a.out /mnt/home/kochocki/egen_lite/plot/get_dom_timing_statistics.py 'ftp_v2_1_3_25_500' '/mnt/home/kochocki/egen_lite/saved_models/' 1 3 25 500 4

sbatch --array=1 -A deyoungbuyin --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/plot_time_stats_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/plot_time_stats_%a.out /mnt/home/kochocki/egen_lite/plot/plot_dom_timing_statistics.py 'ftp_v2_1_3_25_500'
