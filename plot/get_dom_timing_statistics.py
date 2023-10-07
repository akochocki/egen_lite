#!/usr/bin/env python3

from __future__ import division, print_function
import os
import shutil
import logging
import tensorflow as tf
import h5py
import numpy as np

from tensorflow.data import Dataset
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import logging
tf.autograph.set_verbosity(10, alsologtostdout=False)


gpu_devices = tf.config.list_physical_devices('GPU')
#tf.debugging.set_log_device_placement(True)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from icecube import icetray, dataio, dataclasses, phys_services
#from icecube import gulliver, lilliput, gulliver_modules, linefit
from I3Tray import I3Tray
import numpy as np
from I3Tray import *
from glob import glob
from icecube.icetray import I3Units
import matplotlib.pyplot as plt
#from . import cbook
import importlib
from icecube import icetray, dataio, dataclasses, phys_services
from icecube import gulliver, lilliput, gulliver_modules, linefit
from icecube import interfaces, simclasses, sim_services, clsim
from icecube.clsim.traysegments import I3CLSimMakePhotons, I3CLSimMakeHitsFromPhotons
from I3Tray import *
import os,sys
from os.path import expandvars
import logging
import math
from optparse import OptionParser
from configparser import ConfigParser
import numpy as np
from icecube.MuonGun import load_model, Floodlight, StaticSurfaceInjector, Cylinder, OffsetPowerLaw, ExtrudedPolygon
from icecube.hdfwriter import I3HDFWriter
from icecube.MuonGun.segments import GenerateBundles
from icecube.sim_services import I3ParticleTypePropagatorServiceMap
from icecube.PROPOSAL import I3PropagatorServicePROPOSAL
from icecube import DOMLauncher
from icecube.cmc import I3CascadeMCService
from icecube import PROPOSAL
import argparse

# Need to pass in name of model, load weights
# Need to initialize model from kw arguments
# Get MC info of each event in initial set
# Determine model labels given real event, DOM positions
# Use net to predict results
# Compare results to MC truth for event binned on same scale as reconstruction, every 10 ns

geom = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz'
g_file = dataio.I3File(geom)
omgeo = g_file.pop_frame(icetray.I3Frame.Geometry)['I3Geometry'].omgeo
set = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
file = '/mnt/scratch/kochocki/ftp_electrons/set_1/Gen1k_10TeV_EMinus_FTP_V2_1.i3' #start at 1001

# i3tray to open file
# look at inicepulses
# look at predicted event

# for now, just store [  real_zen, real_az, vertex_x, vertex_y, vertex_z, [ real dom, string, posx, posy, posz, true charge, predicted charge, true - predicted charge prediction, true/predicted,  real charge waveform, predicted charge waveform, sum of abs differences ] ]

# should also look at sample waveforms for events, create and bin waveforms/expectations, sum up abs value of differences per bin
# make histogram with event absolute value differences
# for events with largest absolute value differences (apply some cut), look at
# waveforms with largest differences
# difference per DOM as a function of true charge

def IntegrateTimeBin(bin_start, bin_end, mean, std, r, amp):

    if ( (bin_start < mean) and ( bin_end <= mean) ):
        return (amp/(r + 1.0))*(math.erf( (np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_start )/(2.0*std) ) - math.erf( ( np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_end   )/(2.0*std) )  )
    elif ( (bin_start <= mean) and ( mean <= bin_end  )  ):
        left_half = (amp/(r + 1.0))*(math.erf( (np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_start )/(2.0*std) ) - math.erf( ( np.sqrt(2.0 )*mean - np.sqrt(2.0 )*mean   )/(2.0*std) )  )
        right_half =  r*(amp/(r + 1.0))*(math.erf( (np.sqrt(2.0 )*mean - np.sqrt(2.0 )*mean )/(2.0*std*r) ) - math.erf( ( np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_end   )/(2.0*std*r) )  )
        return left_half + right_half;
    else:
        return r*(amp/(r + 1.0))*(math.erf( (np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_start )/(2.0*std*r) ) - math.erf( ( np.sqrt(2.0 )*mean - np.sqrt(2.0 )*bin_end   )/(2.0*std*r) )  )

    
model_str = str(sys.argv[1])
model_save_location = str(sys.argv[2])
num_local_layers = int(sys.argv[3])
num_conv_layers = int(sys.argv[4])
num_local_filters = int(sys.argv[5])
num_conv_filters = int(sys.argv[6])
num_gaussians = int(sys.argv[7])


class Generator(Model):
  def __init__(self):
    super(Generator, self).__init__()
    
    layer_list = []
    
    layer_list.append(layers.Input(shape=(86*60, 11) )
    for i in range(len( num_local_layers)):
        layer_list.append(layers.LocallyConnected1D(filters=num_local_filters, kernel_size=1, strides=1, activation='elu'))
    for i in range(len( num_conv_layers)):
        layer_list.append(layers.Conv1D(filters=num_conv_filters, kernel_size=1, activation='elu', strides=1 ))
    layer_list.append(layers.Conv1D(filters=4*model_number_guassians, kernel_size=1, activation='elu', strides=1 )) # Code assumes 6 Gausians now
          
    self.generator_model = tf.keras.Sequential(layer_list)
  def call(self, x):
    generated = self.generator_model(x)
    return generated
    
try:
    generator = tf.keras.models.load_model(model_save_location + model_str)

except:
    print('Unable to load model!')


real_pred_events = []

detector_coords = np.load('om_position_array.npy')

n_ice_group = 1.35634
c = 0.299792458/n_ice_group # m/ns

def add_event( frame):

    #print(frame['I3EventHeader'] )
    if (frame['I3EventHeader'].sub_event_stream == 'StochasticsEventStream_MCPE_to_Reco') and (len(frame['I3MCPESeriesMap']) > 0) and (set*100 <= frame['I3EventHeader'].event_id) and (frame['I3EventHeader'].event_id < (set + 1)*100)  :
        
        print( 'Processing Event...' )
        
        event_labels = []
    
        real_zen = 0.0
        real_az =  0.0
        real_t =  0.0
        vertex_x =  0.0
        vertex_y =  0.0
        vertex_z =  0.0
        real_energy =  0.0
        loss =  0.0
        
        tmin = -500.0
        tmax = 4500.0
        
        time_bins = np.linspace(tmin, tmax, num=250)
        time_bins_mids = []
        for k in range(len( time_bins) - 1):
            time_bins_mids.append((time_bins[k] + time_bins[k+1])/2.0 )

        
        event_pulse_info = []
        
        real_energy = frame['I3MCTree'][0].energy
        vertex_x = frame['I3MCTree'][0].pos.x
        vertex_y = frame['I3MCTree'][0].pos.y
        vertex_z = frame['I3MCTree'][0].pos.z
        real_zen = frame['I3MCTree'][0].dir.zenith
        real_az = frame['I3MCTree'][0].dir.azimuth
        real_t =  frame['I3MCTree'][0].time
        
        event_dir_x = np.cos(real_az )*np.sin(real_zen) # Unit vectors, particle origin
        event_dir_y = np.sin(real_az )*np.sin(real_zen)
        event_dir_z = np.cos(real_zen)
        
        event_head_x = -1.0*np.cos(real_az )*np.sin(real_zen) # Unit vectors, particle heading
        event_head_y = -1.0*np.sin(real_az )*np.sin(real_zen)
        event_head_z = -1.0*np.cos(real_zen)

        
        for l in range(86):
            for k in range(60):
                om_x, om_y, om_z = detector_coords[l][k]
                
                dist = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )
                dx = (om_x - vertex_x)/dist
                dy = (om_y - vertex_y)/dist
                dz = (om_z - vertex_z)/dist
                
                opening_angle = math.acos(  (event_head_x*dx + event_head_y*dy + event_head_z*dz )/( np.sqrt(dx**2 + dy**2 + dz**2)*np.sqrt(event_head_x**2 + event_head_y**2 + event_head_z**2) )      )
                
                event_labels.append( [vertex_x/600.0, vertex_y/600.0, vertex_z/600.0, event_dir_x, event_dir_y, event_dir_z, dist/600.0, dx, dy, dz, opening_angle ])


        
        
        inice_pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, "SplitInIcePulses")

        labels =  tf.constant(event_labels, dtype=tf.float32)
        result_tensors = generator.predict(labels) # returned
        pulse_info = tf.reshape(result_tensors,  [ 86*60,4*num_gaussians] ) # Need to split to [86*60, num_gaussians]
        
        # amplitude, mean, sigma, r
        
        # Check below transformation
        latent_var_mu = np.array( pulse_info[:,2::4] , dtype=np.double)
        latent_var_sigma = np.array( pulse_info[:,3::4] , dtype=np.double)
        latent_var_r = np.array( pulse_info[:,4::4] , dtype=np.double)
        latent_var_scale = np.array( pulse_info[:,::4] , dtype=np.double)

        
        #need to take predictions per string, om, and compare to potential real set of pulses on same string, om
           

        for dom_entry in range( 86*60 ):
            j = int(dom_entry/86.0) # Every 86 doms, at a new string
            k = dom_entry%86
            string = j + 1
            om = k + 1
            pulse_x, pulse_y, pulse_z = omgeo[OMKey(string, om )].position
            om_x, om_y, om_z = detector_coords[j][k]
           
            propTime = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )/c
            
            pdf_array = []
            for gauss in range(len( latent_var_scale[0] ) ):
                pdf_array.append( [ latent_var_mu[dom_entry][gauss], latent_var_sigma[dom_entry][gauss], latent_var_r[dom_entry][gauss], latent_var_scale[dom_entry][gauss]     ]   )
                
            
            
            pred_binned_charges = []
            pred_charge = 0.0
            charge = 0.0
            
            # assumes all loss interaction vertices at t = 0, no pulses/light observed prior
            #bin_start, bin_end, mean, std, r, amp
            for i in range( len(time_bins) - 1  ):
                bin_sum = 0.0
                for z in range(len( latent_var_sigma[dom_entry] ) ):
                    if time_bins[i] > 0.0:
                        bin_sum = bin_sum + 100.0*IntegrateTimeBin( np.sqrt(time_bins[i] - propTime)/10.0, np.sqrt(time_bins[i + 1] - propTime)/10.0, latent_var_mu[dom_entry][z], latent_var_sigma[dom_entry][z], latent_var_r[dom_entry][z], latent_var_scale[dom_entry][z] )
                    elif time_bins[i + 1] > 0.0 and time_bins[i] < 0.0:
                        bin_sum = bin_sum + IntegrateTimeBin( 0.0, np.sqrt(time_bins[i + 1] - propTime)/10.0, latent_var_mu[dom_entry][z], latent_var_sigma[dom_entry][z], latent_var_r[dom_entry][z], latent_var_scale[dom_entry][z] )
                    else:
                        bin_sum = bin_sum + 0.0
                        
                pred_binned_charges.append( bin_sum)
                
            for z in range(len( latent_var_sigma[dom_entry] ) ):
                charge = charge + IntegrateTimeBin( 0.0, np.sqrt(time_bins[i + 1])/10.0, latent_var_mu[dom_entry][z], latent_var_sigma[dom_entry][z], latent_var_r[dom_entry][z], latent_var_scale[dom_entry][z] )

            
            total_om_charge = 0.0
            dom_hits = []

            for i in inice_pulse_map:
                dom_info = i[0]
                real_string = dom_info.string
                real_om = dom_info.om
                if int( real_om) == int(om) and int( real_string) == int(string):
                    pulse_info = i[1]
                    for z in range( len( pulse_info) ):
                        total_om_charge = total_om_charge + pulse_info[z].charge
                        dom_hits.append([ pulse_info[z].time,  pulse_info[z].charge  ])
                        
            real_pulse_times = []
            real_pulse_charges = []
            
            for i in range( len( dom_hits) ):
                real_pulse_times.append( dom_hits[i][0] )
                real_pulse_charges.append( dom_hits[i][1] )

            
            #### can now determine real and pred waveforms per optical module ####
            
            print('real_pulse_times', real_pulse_times)
            print('real_pulse_charges', real_pulse_charges)
            
            real_binned_charges = []
            for i in range( len(time_bins) - 1  ):
                real_binned_charge = 0.0
                for z in range( len(real_pulse_charges)):
                    if (time_bins[i] <= real_pulse_times[z]) and (real_pulse_times[z] < time_bins[i+1]):
                        real_binned_charge = real_binned_charge + real_pulse_charges[z]
                real_binned_charges.append( real_binned_charge)
            
            
            waveform_abs_dif = 0.0
            for i in range(len(pred_binned_charges)):
                waveform_abs_dif = waveform_abs_dif  + np.abs( real_binned_charges[i] - pred_binned_charges[i] )
            
            
            event_pulse_info.append( [string, om, pulse_x, pulse_y, pulse_z, total_om_charge, charge, total_om_charge - charge, total_om_charge/charge , waveform_abs_dif, real_binned_charges, pred_binned_charges  ])
    
    
        real_pred_events.append(  [ real_zen, real_az, vertex_x,vertex_y,vertex_z,  event_pulse_info ] )

tray = I3Tray()
tray.Add("I3Reader", "reader", FileName = file)
tray.Add( add_event, 'add_event', Streams=[icetray.I3Frame.Physics])
tray.Execute()


np.save( '/mnt/scratch/kochocki/ftp_electrons/set_1/dom_time_stats_EMinus_' + str(1) + '_' + str(set) + '_' + model_str , real_pred_events)

