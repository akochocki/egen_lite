#!/bin/bash

module purge
module load CUDA/11.4.2

PYTHONPATH=/mnt/home/kochocki/prefix/bin:/mnt/home/kochocki/prefix/lib/python3.7/site-packages:/mnt/home/kochocki/prefix/lib/python3.6/site-packages:/mnt/home/kochocki/nuSQuIDS_build/resources/python/bindings:/mnt/home/kochocki/prefix:/mnt/home/kochocki/prefix/lib64/python3.6/site-packages

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`

PYTHONPATH=/mnt/home/kochocki/prefix/lib64/:$PYTHONPATH
LD_LIBRARY_PATH=/mnt/home/kochocki/prefix/lib64/:$LD_LIBRARY_PATH

#--gres=gpu:v100:1
#--gres=gpu:a100:1 
 
sbatch --array=2 -A deyoungbuyin --gres=gpu:1 --gres=gpu:v100:1 --mem=32000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_1_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_1_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_25_500' 1 3 25 500 # original

sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --gres=gpu:v100:1 --mem=32000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_1_6p_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_1_6p_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_6g.py 'loss_model_ftp_v2_6p_1_3_25_500' 1 3 25 500 # original




#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_2_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_2_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_3_9_25_500' 3 9 25 500 # increase depth

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_3_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_3_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_50_1000' 1 3 50 1000 # increase width

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_4_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_4_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_3_3_25_500' 3 3 25 500 # increase depth for connected layer

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_5_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_5_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_50_500' 1 3 50 500 # increase width for connected layer

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_6_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_6_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_9_500' 1 3 9 500 # increase depth for conv layer

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_7_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_7_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_25_1000' 1 3 25 1000 # increase width for conv layer


#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_8_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_8_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_12_250' 1 3 12 250 # original, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_9_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_9_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_25_250' 1 3 25 250 # maintain connected width, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_10_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_10_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_12_500' 1 3 12 500 # maintain conv width, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_11_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_11_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_3_9_12_250' 3 9 12 250 # increase depth, decrease width, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_12_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_12_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_3_3_12_250' 3 3 12 250 # increase depth for connected layer, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_13_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_13_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_50_250' 1 3 50 250 # increase width for connected layer, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_14_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_14_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_9_12_250' 1 9 12 250 # increase depth for conv layer, decrease width otherwise

#sbatch --array=1 -A deyoungbuyin --gres=gpu:1 --mem=28000 --time=72:00:00 --error=/mnt/home/kochocki/egen_lite/logs/train_model_15_%a.err --output=/mnt/home/kochocki/egen_lite/logs/train_model_15_%a.out /mnt/home/kochocki/egen_lite/train/train_generator_4g.py 'loss_model_ftp_v2_1_3_12_1000' 1 3 12 1000 # increase width for conv layer, decrease width otherwise

