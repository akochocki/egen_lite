module purge
module load CUDA/11.4.2
                                                                    
PYTHONPATH=/mnt/home/kochocki/prefix/bin:/mnt/home/kochocki/prefix/lib/python3.7/site-packages:/mnt/home/kochocki/prefix/lib/python3.6/site-packages:/mnt/home/kochocki/nuSQuIDS_build/resources/python/bindings:/mnt/home/kochocki/prefix:/mnt/home/kochocki/prefix/lib64/python3.6/site-packages
                                                                   
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
                                                                                                                                                                                
PYTHONPATH=/mnt/home/kochocki/prefix/lib64/:$PYTHONPATH
LD_LIBRARY_PATH=/mnt/home/kochocki/prefix/lib64:/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/lib:/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/lib64:$LD_LIBRARY_PATH
                                                                                                                                                                            
/mnt/home/kochocki/egen_lite/icetray_build/env-shell.sh

######### Decoder Reco NuE CC #########
sbatch --array=1-250 -A deyoungbuyin  --gres=gpu:1  --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/AutoVar_Reco_NuE_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/AutoVarReco_NuE_%a.out /mnt/home/kochocki/egen_lite/reconstruct/event_reconstruction_monopod_varauto.py -s 'NuE' -m 'loss_model_ftp_v2_1_3_25_500'

######### Table Reco NuE CC #########
sbatch --array=1-250 -A deyoungbuyin  --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/Table_Reco_NuE_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/Table_Reco_NuE_%a.out /mnt/home/kochocki/egen_lite/reconstruct/event_reconstruction_monopod_table.py -s 'NuE'

######### Decoder Reco MuonGun #########
sbatch --array=1-250 -A deyoungbuyin --gres=gpu:1 --mem=12000 --time=96:00:00 --error=/mnt/home/kochocki/egen_lite/logs/AutoVar_Reco_MuonGun_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/AutoVar_Reco_MuonGun_%a.out /mnt/home/kochocki/egen_lite/reconstruct/event_reconstruction_millipede_varauto.py -s 'MuonGun' -m 'loss_model_ftp_v2_1_3_25_500'

######### Table Reco MuonGun #########
sbatch --array=1-250 -A deyoungbuyin  --mem=24000 --time=48:00:00 --error=/mnt/home/kochocki/egen_lite/logs/Table_Reco_MuonGun_%a.err  --output=/mnt/home/kochocki/egen_lite/logs/Table_Reco_MuonGun_%a.out /mnt/home/kochocki/egen_lite/reconstruct/event_reconstruction_millipede_table.py -s 'MuonGun'

