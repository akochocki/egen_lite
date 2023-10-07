#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import math
import glob
from datetime import datetime
import time
import os,sys
from os.path import expandvars


from tensorflow.data import Dataset
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import logging
tf.autograph.set_verbosity(10, alsologtostdout=False)


gpu_devices = tf.config.list_physical_devices('GPU')
#tf.debugging.set_log_device_placement(True)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float64')

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

n_ice_group = 1.35634
c = 0.299792458/n_ice_group # m/ns


detector_coords = np.load('om_position_array.npy') # 86 * 60 * 3

def get_data_set(set_size, set_location, file_set ):

    # x should be 86*60 DOM input label sets per event, input and training data as vectors of this size (train per event!), with empty sets for DOMs w/o hits

    event_count = 0
    
    event_DOM_hypothesis = [] # set_size * 86*60 * [vertex_x, vertex_y, vertex_z, event_dir_x, event_dir_y, event_dir_z, vertex_dist_to_DOM, dx, dy, dz, opening_angle (event dir and dir to DOM)]
    pulse_series_combined = []
    
    # list of events, charge and time list per all optical module with varying length
    
    # For each new event, prefill event_DOM_hypothesis in order
    # Create a list of empty pulse series sets, iterate over observed pulses to fill in
    
    # For all events, get maximum pulse series length
    # Reiterate over all events, fill in with 0s (skip these in loss function)

    for file_int in range(set_size):
        file_str = set_location + 'Gen1k_10TeV_EMinus_FTP_V2_' + str( set_size*(file_set - 1) + file_int + 1 ) + '.hdf5'
        try:
            with h5py.File(file_str,'r') as file:
                pulses = np.array(file['SplitInIcePulses']) # Ordered by event ID
                event_hypothesis = np.array(file['LabelsDeepLearning'])
                # for each event, event ID, gather pulse series per optical module
                times = []
                charges = []
                current_event_ID = pulses[0][1]
                current_om = pulses[0][6]
                current_string = pulses[0][5]
                     
                event_labels = [] # Need to set for each new event
                
                event_pulse_series_times = []
                event_pulse_series_charges = []  # Fill with [om, charge]
                
                
                for j in range(len(event_hypothesis )):
                    if event_hypothesis[j][1] == current_event_ID:
                        vertex_x = event_hypothesis[j][5]
                        vertex_y = event_hypothesis[j][6]
                        vertex_z = event_hypothesis[j][7]
                        event_dir_azimuth = event_hypothesis[j][10] # radians, direction particle comes from
                        event_dir_zenith = event_hypothesis[j][9] # radians
                        event_dir_x = np.cos(event_dir_azimuth )*np.sin(event_dir_zenith) # Unit vectors, particle origin
                        event_dir_y = np.sin(event_dir_azimuth )*np.sin(event_dir_zenith)
                        event_dir_z = np.cos(event_dir_zenith)
                        
                        event_head_x = -1.0*np.cos(event_dir_azimuth )*np.sin(event_dir_zenith) # Unit vectors, particle heading
                        event_head_y = -1.0*np.sin(event_dir_azimuth )*np.sin(event_dir_zenith)
                        event_head_z = -1.0*np.cos(event_dir_zenith)

                        for l in range(86):
                            for k in range(60):
                                om_x, om_y, om_z = detector_coords[l][k]
                                
                                dist = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )
                                dx = (om_x - vertex_x)/dist
                                dy = (om_y - vertex_y)/dist
                                dz = (om_z - vertex_z)/dist
                                
                                opening_angle = math.acos(  (event_head_x*dx + event_head_y*dy + event_head_z*dz )/( np.sqrt(dx**2 + dy**2 + dz**2)*np.sqrt(event_head_x**2 + event_head_y**2 + event_head_z**2) )      )
                                
                                event_labels.append( [vertex_x/600.0, vertex_y/600.0, vertex_z/600.0, event_dir_x, event_dir_y, event_dir_z, dist/600.0, dx, dy, dz, opening_angle ])
                                event_pulse_series_times.append( [ 0.0 ])
                                event_pulse_series_charges.append( [ 0.0 ])


                
                for i in range(len(pulses)):
            
                    event_ID = pulses[i][1]
                    pulse_om = pulses[i][6]
                    pulse_string = pulses[i][5]
                    
                    # For each pulse corresponding to a new event_ID, build up event_labels array for all DOMs and that interaction vertex, create new pulse set with empty arrays
                    # Iterate over rest of pulses, fill into event arrays
                    
               
                    if not ( ( event_ID == current_event_ID) and (pulse_om == current_om) and (pulse_string == current_string) ): # Either new event or new optical module (event ID reset every file)
                    
                        if event_ID == current_event_ID: # Same event, new optical module
                    
                            # Shift all events by propagation time from vertex to OM
                            om_x, om_y, om_z = detector_coords[current_string - 1][current_om - 1]
                            propTime = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )/c
                            event_pulse_series_times[60*(current_string - 1) + current_om - 1] = (times - np.full(len(times), propTime)).tolist()
                            event_pulse_series_charges[60*(current_string - 1) + current_om - 1] = charges
                                                    
                            times = []
                            charges = []
                            
                            current_event_ID = event_ID
                            current_om = pulse_om
                            current_string = pulse_string
                        
                        else: # New event ID
                        
                            # Shift all events by propagation time from vertex to OM
                            om_x, om_y, om_z = detector_coords[current_string - 1][current_om - 1]
                            propTime = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )/c
                            event_pulse_series_times[60*(current_string - 1) + current_om - 1] = (times - np.full(len(times), propTime)).tolist()
                            event_pulse_series_charges[60*(current_string - 1) + current_om - 1] = charges
                         
                            event_DOM_hypothesis.append(event_labels)
                            
                            pulse_series_combined.append( [ event_pulse_series_times, event_pulse_series_charges ]  )
                            
                            event_labels = []
                            event_pulse_series_times = []
                            event_pulse_series_charges = []
                            event_count = event_count + 1
                            times = []
                            charges = []
                            
                            current_event_ID = event_ID
                            current_om = pulse_om
                            current_string = pulse_string
                            
                            for j in range(len(event_hypothesis )):
                                if event_hypothesis[j][1] == current_event_ID:
                                    vertex_x = event_hypothesis[j][5]
                                    vertex_y = event_hypothesis[j][6]
                                    vertex_z = event_hypothesis[j][7]
                                    event_dir_azimuth = event_hypothesis[j][10] # radians, direction particle comes from
                                    event_dir_zenith = event_hypothesis[j][9] # radians
                                    event_dir_x = np.cos(event_dir_azimuth )*np.sin(event_dir_zenith) # Unit vectors, particle origin
                                    event_dir_y = np.sin(event_dir_azimuth )*np.sin(event_dir_zenith)
                                    event_dir_z = np.cos(event_dir_zenith)
                                    
                                    event_head_x = -1.0*np.cos(event_dir_azimuth )*np.sin(event_dir_zenith) # Unit vectors, particle heading
                                    event_head_y = -1.0*np.sin(event_dir_azimuth )*np.sin(event_dir_zenith)
                                    event_head_z = -1.0*np.cos(event_dir_zenith)

                                    for l in range(86):
                                        for k in range(60):
                                            om_x, om_y, om_z = detector_coords[l][k]
                                            
                                            dist = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )
                                            dx = (om_x - vertex_x)/dist
                                            dy = (om_y - vertex_y)/dist
                                            dz = (om_z - vertex_z)/dist
                                            
                                            opening_angle = math.acos(  (event_head_x*dx + event_head_y*dy + event_head_z*dz )/( np.sqrt(dx**2 + dy**2 + dz**2)*np.sqrt(event_head_x**2 + event_head_y**2 + event_head_z**2) )      )
                                            
                                            event_labels.append( [vertex_x/600.0, vertex_y/600.0, vertex_z/600.0, event_dir_x, event_dir_y, event_dir_z, dist/600.0, dx, dy, dz, opening_angle ])
                                            event_pulse_series_times.append( [ 0.0 ])
                                            event_pulse_series_charges.append( [ 0.0 ])

                
                                
                
                    
                    
                    pulse_time = pulses[i][9]
                    pulse_charge = pulses[i][11]
                    
                    if (event_ID == current_event_ID) and (pulse_om == current_om) and (pulse_string == current_string):
                        times.append( pulse_time)
                        charges.append( pulse_charge)
                
                    if i == (len(pulses) - 1):
                        
                        # Shift all events by propagation time from vertex to OM
                        om_x, om_y, om_z = detector_coords[current_string - 1][current_om - 1]
                        propTime = np.sqrt( (vertex_x - om_x)**2 + (vertex_y - om_y)**2 + (vertex_z - om_z)**2 )/c
                        event_pulse_series_times[60*(current_string - 1) + current_om - 1] = (times  - np.full(len(times), propTime)).tolist()
                        event_pulse_series_charges[60*(current_string - 1) + current_om - 1] = charges
                        
                        pulse_series_combined.append( [ event_pulse_series_times, event_pulse_series_charges ]  )
                                            
                        event_pulse_series_times = []
                        event_pulse_series_charges = []
                        
                        times = []
                        charges = []
                    
                        event_DOM_hypothesis.append(event_labels)

                        event_count = event_count + 1
        
        except Exception as e:
            print(e)
        
        
    print(len(event_DOM_hypothesis[4])  )
    print(len(pulse_series_combined[4]), len(pulse_series_combined[4][1] ))
    
    event_DOM_hypothesis_as_tensor = tf.convert_to_tensor(event_DOM_hypothesis, dtype=tf.double)
    pulse_series_combined_as_tensor = tf.ragged.constant(pulse_series_combined, dtype=tf.double )

    return [event_DOM_hypothesis_as_tensor,pulse_series_combined_as_tensor]
    


def serialize_example(x, y):
    data = {
            "f": _bytes_feature(serialize_array( x.merge_dims(0, -1)  )), # flat_pulse_data
            "d":  _bytes_feature(serialize_array( np.asarray(x.nested_row_lengths()[0]) )), # pulse_data_dims
            "r": _bytes_feature(serialize_array( np.asarray(x.nested_row_lengths()[1]) )), # pulse_data_ragged_dims
            "l":  _bytes_feature(serialize_array(y))  } # labels
    example = tf.train.Example(features=tf.train.Features(feature=data))
    return example.SerializeToString()


# Will store every ten hdf5 files within a TFRecord of ~1 GB (~4000 TFRecords goal)

set = int(os.getenv('SLURM_ARRAY_TASK_ID'))

folder = 'set_' + str(int(float(set*5 - 1)/1000.0)) # sets 1-100 in first folder
training_event_DOM_hypothesis, training_pulse_series = get_data_set(5, '/mnt/scratch/kochocki/ftp_electrons/' + folder + '/', set)

path = '/mnt/scratch/kochocki/ftp_electrons/' + folder + '/' + 'record_' + str(set) + '.tfrecords'
compression_type='GZIP'
options = tf.io.TFRecordOptions(compression_type=compression_type)

dataset = tf.data.Dataset.from_tensor_slices((training_pulse_series, training_event_DOM_hypothesis))
with tf.io.TFRecordWriter(path, options) as writer:
    for i, (x_instance, y_instance) in enumerate(dataset):
        writer.write(serialize_example(x_instance, y_instance ))



