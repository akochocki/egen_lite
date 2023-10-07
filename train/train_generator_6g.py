import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import math
import glob
from datetime import datetime
import time
import glob
import os, sys
import matplotlib as mpl

from tensorflow.data import Dataset
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import logging

tf.autograph.set_verbosity(10, alsologtostdout=False)
#tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx('float64')
#print(tf.keras.backend.floatx()) # Check precision set properly


gpu_devices = tf.config.list_physical_devices('GPU')
#tf.debugging.set_log_device_placement(True)
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def format_plot(n):
    fig = plt.figure(n)
    plt.rc('font',family='serif') #setting font style for entire plot and its objects
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['mathtext.it'] = 'serif:italic'
    mpl.rcParams['mathtext.bf'] = 'serif:bold' #change font of math text to serif
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1 #also set size of border line
    ax = plt.subplot(111) # ax is an object of of the figure we're making
    ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=1, direction='in', right='on')
    return ax

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

def parse_tfr_element(element):

    data = {
        "f": tf.io.FixedLenFeature([], tf.string), # flattened pulse data
        "d":  tf.io.FixedLenFeature([], tf.string), # dimension of pulse data
        "r": tf.io.FixedLenFeature([], tf.string), # ragged dimension of pulse data
        "l":  tf.io.FixedLenFeature([], tf.string)  } # 5160 * 11 (or vector length of labels)

    content = tf.io.parse_single_example(element, data)

    flat_pulse_data = content["f"]
    pulse_data_dims = content["d"]
    pulse_data_ragged_dims = content["r"]
    labels = content["l"]
    

    # create a vector of the value 20.0, flat
    # map to raggged dimension

    flat_pulse_data = tf.io.parse_tensor(flat_pulse_data, out_type=tf.double) # double precision pulse data
    pulse_data_dims = tf.io.parse_tensor(pulse_data_dims, out_type=tf.int64)
    pulse_data_ragged_dims = tf.io.parse_tensor(pulse_data_ragged_dims, out_type=tf.int64)
    labels = tf.io.parse_tensor(labels, out_type=tf.double) # double precision labels

    tf.debugging.check_numerics(flat_pulse_data, "Invalid tensor" ) # check for nans, infs
    tf.debugging.check_numerics(labels, "Invalid tensor" )

    # first half of string is timing info
    # second half is charge info
    timing_info, charge_info = tf.split(flat_pulse_data,2,axis=0) # split into timing, charge info
    
    transformed_timing_info = tf.math.sqrt(tf.maximum(timing_info + 20.0, 0.0))/10.0 # trasnform our timing info for a better fit. Prepulses up to 20 ns before the interaction are kept
    transformed_charge_info = charge_info/100.0 # trasnform our charge info for a better fit
    
    ragged_pulse_data = tf.RaggedTensor.from_row_lengths(
        values=tf.concat([transformed_timing_info, transformed_charge_info], 0),
        row_lengths=pulse_data_ragged_dims) # Add back ragged structure

    pulse_data = tf.RaggedTensor.from_uniform_row_length(ragged_pulse_data, pulse_data_dims[0]) # Assumes a row length like 60*86, creating two ragged rows of this length (time, charge). Should now be a 60*86 * 2 * ragged array
    
    return (labels, pulse_data) # return our training data and target data (x, y)
    

    
def tfrecords_reader_dataset(bp, batch_size=100, shuffle_buffer_size=10000, n_readers=6):
    dataset = tf.data.Dataset.list_files(bp, shuffle=True)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
        cycle_length=n_readers,
        num_parallel_calls=tf.data.AUTOTUNE, # opt
        deterministic=False) # opt
    if shuffle_buffer_size > 0:
        dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_tfr_element,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.AUTOTUNE)


# Below, our functions for the Gaussian mixture model
sqrt2 = tf.cast( tf.math.sqrt(2.0*np.pi), tf.double)

def asymm_gaussian_6(x, A1, mu1, sigma1, r1, A2, mu2, sigma2, r2, A3, mu3, sigma3, r3, A4, mu4, sigma4, r4, A5, mu5, sigma5, r5, A6, mu6, sigma6, r6):
    
    ''' Below is an old version of the function useful for inspection
    prefactor1 = A1*(2.0/(sqrt2*sigma1*(r1 + 1.0)))
    prefactor2 = A2*(2.0/(sqrt2*sigma2*(r2 + 1.0)))
    prefactor3 = A3*(2.0/(sqrt2*sigma3*(r3 + 1.0)))
    prefactor4 = A4*(2.0/(sqrt2*sigma4*(r4 + 1.0)))
    prefactor5 = A5*(2.0/(sqrt2*sigma5*(r5 + 1.0)))
    prefactor6 = A6*(2.0/(sqrt2*sigma6*(r6 + 1.0)))
    
    less_than_1 = tf.where(x <= mu1, 1.0, 0.0)
    greater_than_1 = tf.where(x > mu1, 1.0, 0.0)
    less_than_2 = tf.where(x <= mu2, 1.0, 0.0)
    greater_than_2 = tf.where(x > mu2, 1.0, 0.0)
    less_than_3 = tf.where(x <= mu3, 1.0, 0.0)
    greater_than_3 = tf.where(x > mu3, 1.0, 0.0)
    less_than_4 = tf.where(x <= mu4, 1.0, 0.0)
    greater_than_4 = tf.where(x > mu4, 1.0, 0.0)
    less_than_5 = tf.where(x <= mu5, 1.0, 0.0)
    greater_than_5 = tf.where(x > mu5, 1.0, 0.0)
    less_than_6 = tf.where(x <= mu6, 1.0, 0.0)
    greater_than_6 = tf.where(x > mu6, 1.0, 0.0)

    sum_less_than_1 = prefactor1*tf.math.exp( -1.0*((x - mu1)**2)/(2*sigma1**2) )
    sum_greater_than_1 = prefactor1*tf.math.exp( -1.0*((x - mu1)**2)/(2*(sigma1*r1)**2) )
    sum_less_than_2 = prefactor2*tf.math.exp( -1.0*((x - mu2)**2)/(2*sigma2**2) )
    sum_greater_than_2 = prefactor2*tf.math.exp( -1.0*((x - mu2)**2)/(2*(sigma2*r2)**2) )
    sum_less_than_3 = prefactor3*tf.math.exp( -1.0*((x - mu3)**2)/(2*sigma3**2) )
    sum_greater_than_3 = prefactor3*tf.math.exp( -1.0*((x - mu3)**2)/(2*(sigma3*r3)**2) )
    sum_less_than_4 = prefactor4*tf.math.exp( -1.0*((x - mu4)**2)/(2*sigma4**2) )
    sum_greater_than_4 = prefactor4*tf.math.exp( -1.0*((x - mu4)**2)/(2*(sigma4*r4)**2) )
    sum_less_than_5 = prefactor5*tf.math.exp( -1.0*((x - mu5)**2)/(2*sigma5**2) )
    sum_greater_than_5 = prefactor5*tf.math.exp( -1.0*((x - mu5)**2)/(2*(sigma5*r5)**2) )
    sum_less_than_6 = prefactor6*tf.math.exp( -1.0*((x - mu6)**2)/(2*sigma6**2) )
    sum_greater_than_6 = prefactor6*tf.math.exp( -1.0*((x - mu6)**2)/(2*(sigma6*r6)**2) )

    sum = less_than_1*sum_less_than_1 + greater_than_1*sum_greater_than_1 + less_than_2*sum_less_than_2 + greater_than_2*sum_greater_than_2 + less_than_3*sum_less_than_3 + greater_than_3*sum_greater_than_3 + less_than_4*sum_less_than_4 + greater_than_4*sum_greater_than_4 + less_than_5*sum_less_than_5 + greater_than_5*sum_greater_than_5 + less_than_6*sum_less_than_6 + greater_than_6*sum_greater_than_6
    '''
    
    # The sum is the predicted value of the mixture model PDF at the given pulse time. A linear combination of asymmetric gaussians.
    sum = tf.cast( tf.where(x <= mu1, 1.0, 0.0), dtype=tf.float64)*A1*(2.0/(sqrt2*sigma1*(r1 + 1.0)))*tf.math.exp( -1.0*((x - mu1)**2)/(2*sigma1**2) ) + tf.cast( tf.where(x > mu1, 1.0, 0.0), dtype=tf.float64)*A1*(2.0/(sqrt2*sigma1*(r1 + 1.0)))*tf.math.exp( -1.0*((x - mu1)**2)/(2*(sigma1*r1)**2) ) + tf.cast( tf.where(x <= mu2, 1.0, 0.0), dtype=tf.float64)*A2*(2.0/(sqrt2*sigma2*(r2 + 1.0)))*tf.math.exp( -1.0*((x - mu2)**2)/(2*sigma2**2) ) + tf.cast( tf.where(x > mu2, 1.0, 0.0), dtype=tf.float64)*A2*(2.0/(sqrt2*sigma2*(r2 + 1.0)))*tf.math.exp( -1.0*((x - mu2)**2)/(2*(sigma2*r2)**2) ) + tf.cast( tf.where(x <= mu3, 1.0, 0.0), dtype=tf.float64)*A3*(2.0/(sqrt2*sigma3*(r3 + 1.0)))*tf.math.exp( -1.0*((x - mu3)**2)/(2*sigma3**2) ) +  tf.cast( tf.where(x > mu3, 1.0, 0.0), dtype=tf.float64)*A3*(2.0/(sqrt2*sigma3*(r3 + 1.0)))*tf.math.exp( -1.0*((x - mu3)**2)/(2*(sigma3*r3)**2) ) + tf.cast( tf.where(x <= mu4, 1.0, 0.0), dtype=tf.float64)*A4*(2.0/(sqrt2*sigma4*(r4 + 1.0)))*tf.math.exp( -1.0*((x - mu4)**2)/(2*sigma4**2) ) + tf.cast( tf.where(x > mu4, 1.0, 0.0), dtype=tf.float64)*A4*(2.0/(sqrt2*sigma4*(r4 + 1.0)))*tf.math.exp( -1.0*((x - mu4)**2)/(2*(sigma4*r4)**2) )  + tf.cast( tf.where(x <= mu5, 1.0, 0.0), dtype=tf.float64)*A5*(2.0/(sqrt2*sigma5*(r5 + 1.0)))*tf.math.exp( -1.0*((x - mu5)**2)/(2*sigma5**2) ) + tf.cast( tf.where(x > mu5, 1.0, 0.0), dtype=tf.float64)*A5*(2.0/(sqrt2*sigma5*(r5 + 1.0)))*tf.math.exp( -1.0*((x - mu5)**2)/(2*(sigma5*r5)**2) ) + tf.cast( tf.where(x <= mu6, 1.0, 0.0), dtype=tf.float64)*A6*(2.0/(sqrt2*sigma6*(r6 + 1.0)))*tf.math.exp( -1.0*((x - mu6)**2)/(2*sigma6**2) ) + tf.cast( tf.where(x > mu6, 1.0, 0.0), dtype=tf.float64)*A6*(2.0/(sqrt2*sigma6*(r6 + 1.0)))*tf.math.exp( -1.0*((x - mu6)**2)/(2*(sigma6*r6)**2) )

    return sum

tf_asymm_gaussian_6 = tf.function(asymm_gaussian_6)


class Unbinned_Pulse_Loss_Ordered(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, real_event_pulses, pred_event_pulse_dist):
        # Input to loss will be of shape [num_passed_events, 86*60]
        # Can check, tf.executing_eagerly()
        
        # Forcing A, mu, sigma, r positive by taking the exponential of the passed values
        
        llh = 0.0

        num_passed_events =  tf.shape(pred_event_pulse_dist)[0] # The dynamic shape of the passed pulse data. Will vary with the selected batch size (number of events)
        
        flattened_pred_event_pulse_dist = tf.reshape( pred_event_pulse_dist, [ 86*60*num_passed_events,24]) # Flatten along the dimension of the number of events to shape [ 86*60*num_passed_events,24]
        
        flattened_pred_event_pulse_dist_transpose =  tf.expand_dims(tf.math.exp(tf.transpose(flattened_pred_event_pulse_dist)) ,  axis=1) # Transpose to [24, 86*60 * num_passed_events], EXPONENTIAL VARIABLE TRANSFORM, expand along last axis
        
        amplitudes = flattened_pred_event_pulse_dist[:,::4] # Result is [num_passed_events * 60*86, all four amplitudes (every fourth parameter)]
        
        amplitudes_reshape = tf.reshape(amplitudes, [num_passed_events, 86*60*4]) # Reshape to have [num_passed_events, 86*60*4 = every four amplitudes for DOM grouped together ]
        
        pred_charge_sum = tf.math.reduce_sum(tf.math.exp(amplitudes_reshape), 1) + 1.0e-7 # Total charge array summed for all passed events, [num_passed_events], EXPONENTIAL VARIABLE TRANSFORM
        
        just_mus = tf.math.exp(flattened_pred_event_pulse_dist[:,2::4]) # [ 86*60*num_passed_events , 4 mus ], EXPONENTIAL VARIABLE TRANSFORM
        
        ordered_indices = tf.argsort(just_mus) # Get indices ordering the value of the mus
        
        ordered_mus = tf.reshape( tf.gather(just_mus, ordered_indices, batch_dims=-1), [num_passed_events*86*60, 4]) # Mus are now in monotonic order
        
        ordered_mus_transpose =  tf.expand_dims( tf.transpose( ordered_mus ) , axis=1) # Reshape ordered mus [4 mus, num_passed_events*86*60]
        
        
        flattened_real_event_pulse_times =   real_event_pulses[:,:1].merge_dims(0,1).merge_dims(0,1)  # Recall that data is transformed when read to make fitting easier. Here, flatten, leave last ragged dimension
        flattened_real_event_pulse_charges = real_event_pulses[:,1:].merge_dims(0,1).merge_dims(0,1)  # Recall that charge is transformed when read. Dividing by 100. Highest charge OM has 30–350 pe @ 10 TeV

        where_zero_flat = tf.not_equal(  flattened_real_event_pulse_charges.merge_dims(0,1), zero) # Mask representing where the flattened, ragged pulse values are zero (a vector)
        ragged_dims = flattened_real_event_pulse_charges.nested_row_lengths()[0] # Get ragged dimensions
        where_zero_ragged = tf.RaggedTensor.from_row_lengths(values=where_zero_flat, row_lengths=ragged_dims).merge_dims(0,1) # Create a ragged mask from the ragged dimensions and flat mask
        where_zero_ragged.set_shape([None]) # Set dynamic mask size
                
        ones = tf.fill( [tf.shape( flattened_real_event_pulse_times.merge_dims(0,1) )[0] ], one) # An array of ones of flattened size
        ones_ragged = tf.RaggedTensor.from_row_lengths( values=ones, row_lengths=ragged_dims) # A ragged array of ones. We will broadcast predicted params to this shape
        
        # Process:
        # list of zeros of size flattened event pulses
        # create ragged zero list based on row splits
        # broadcast predicted parameters based on zero row splits
        # flatten broadcasted params
        # mask pulses and mask params

        # Old call for original implementation
        #pdf_vals = tf_asymm_gaussian_6( flattened_real_event_pulse_times , tf.transpose(flattened_pred_event_pulse_dist_transpose[0]), tf.transpose(ordered_mus_transpose[0]), tf.transpose(flattened_pred_event_pulse_dist_transpose[2]), tf.transpose(flattened_pred_event_pulse_dist_transpose[3]), tf.transpose(flattened_pred_event_pulse_dist_transpose[4]), tf.transpose(ordered_mus_transpose[1]), tf.transpose(flattened_pred_event_pulse_dist_transpose[6]), tf.transpose(flattened_pred_event_pulse_dist_transpose[7]), tf.transpose(flattened_pred_event_pulse_dist_transpose[8]), tf.transpose(ordered_mus_transpose[2]), tf.transpose(flattened_pred_event_pulse_dist_transpose[10]), tf.transpose(flattened_pred_event_pulse_dist_transpose[11]), tf.transpose(flattened_pred_event_pulse_dist_transpose[12]), tf.transpose(ordered_mus_transpose[3]), tf.transpose(flattened_pred_event_pulse_dist_transpose[14]), tf.transpose(flattened_pred_event_pulse_dist_transpose[15]), tf.transpose(flattened_pred_event_pulse_dist_transpose[16]), tf.transpose(ordered_mus_transpose[4]), tf.transpose(flattened_pred_event_pulse_dist_transpose[18]), tf.transpose(flattened_pred_event_pulse_dist_transpose[19]), tf.transpose(flattened_pred_event_pulse_dist_transpose[20]), tf.transpose(ordered_mus_transpose[5]), tf.transpose(flattened_pred_event_pulse_dist_transpose[22]), tf.transpose(flattened_pred_event_pulse_dist_transpose[23]) )

        # The below call evaluates the PDF Gaussian mixture model as a function of time and pulse parameters. Only entries for nonzero charges are considered.


        pdf_vals = tf_asymm_gaussian_4( tf.boolean_mask( flattened_real_event_pulse_times.merge_dims(0,1) , where_zero_ragged ) ,
                tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[0]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                tf.boolean_mask( (tf.transpose(ordered_mus_transpose[0]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[2]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[3]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[4]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(ordered_mus_transpose[1]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[6]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[7]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[8]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(ordered_mus_transpose[2]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[10]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[11]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( (tf.transpose(flattened_pred_event_pulse_dist_transpose[12]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(ordered_mus_transpose[3]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[14]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[15]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[16]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(ordered_mus_transpose[4]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[18]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[19]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[20]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(ordered_mus_transpose[5]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                 tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[22]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged ),
                  tf.boolean_mask( ( tf.transpose(flattened_pred_event_pulse_dist_transpose[23]) * ones_ragged ).merge_dims(0,1) , where_zero_ragged )  )



        # Real pulse charge * log ( predicted charge from PDF )
        pulse_contribs = tf.math.reduce_sum(  tf.boolean_mask( flattened_real_event_pulse_charges.merge_dims(0,1) , where_zero_ragged )*tf.math.log( pdf_vals + 1.0e-7) )
        
        llh = llh + pulse_contribs
        
        llh = llh - pred_charge_sum # Integrate over all gaussians for total predicted charge (all OMs over all events), subtracting from LLH
        return  -1.0*llh # Single value

    

#set = int(os.getenv('SLURM_ARRAY_TASK_ID'))

model_number_guassians = 6
model_save_location = '/mnt/home/kochocki/egen_lite/saved_models/'
model_name = str(sys.argv[1])

num_local_layers = int(sys.argv[1])
num_conv_layers = int(sys.argv[2])
num_local_filters = int(sys.argv[3])
num_conv_filters = int(sys.argv[4])

# Autoencoder or variational autoencoder (would generate means, spreads to describe distributions in hidden layer )
# These are ideally deep generative models (deep networks have many layers as opposed to filters)

class Generator(Model):
  def __init__(self):
    super(Generator, self).__init__()
    
    layer_list = []
    
    layer_list.append(layers.Input(shape=(86*60, 11) ))
    for i in range( num_local_layers):
        layer_list.append(layers.LocallyConnected1D(filters=num_local_filters, kernel_size=1, strides=1, activation='elu'))
    for i in range( num_conv_layers):
        layer_list.append(layers.Conv1D(filters=num_conv_filters, kernel_size=1, activation='elu', strides=1 ))
    layer_list.append(layers.Conv1D(filters=4*model_number_guassians, kernel_size=1, activation='elu', strides=1 )) # Code assumes 6 Gausians now
          
    self.generator_model = tf.keras.Sequential(layer_list)
  def call(self, x):
    generated = self.generator_model(x)
    return generated
    
try:
    generator = tf.keras.models.load_model(model_save_location + model_name)

except:
    generator = Generator()
    generator.compile(optimizer='adam', run_eagerly=True, loss=Unbinned_Pulse_Loss_Ordered()) #run_eagerly=True,
    
    
callback = tf.keras.callbacks.ModelCheckpoint( model_save_location + model_name + '/checkpoint/cp-{epoch:04d}.ckpt', verbose=1, save_weights_only=True, save_freq='epoch')
#backup_and_restore = tf.keras.callbacks.BackupAndRestore( model_save_location + model_name + '/checkpoint/cp-{epoch:04d}.ckpt', save_freq='epoch', delete_checkpoint=True,save_before_preemption=False)
csv = tf.keras.callbacks.CSVLogger( model_save_location + model_name + '_log.csv', separator=',', append=True)
generator.save_weights(model_save_location + model_name + '/checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=0))

    
paths = glob.glob('/mnt/scratch/kochocki/ftp_electrons/set_*/' + 'record_*.tfrecords')
n_files = len(paths)
paths_train = []
paths_validate = []
for i in range(n_files):
    if i%10 == 0:
        paths_validate.append(paths[i]) # Ten percent of all files ares used for validation
    else:
        paths_train.append(paths[i]) # Ten percent of all files ares used for training

print(len(paths_validate), " records used for validation.")
print(len(paths_train), " records used for training.")

batch_size = 100 # This seems to just fit on the MSU HPCC dev node GPUs
shuffle_buffer_size = 5 * batch_size
ds_train = tfrecords_reader_dataset(paths_train, batch_size=batch_size,
                                    shuffle_buffer_size=shuffle_buffer_size,
                                    n_readers=16)
ds_test = tfrecords_reader_dataset(paths_validate, batch_size=batch_size,
                                    shuffle_buffer_size=shuffle_buffer_size,
                                    n_readers=16)


history = generator.fit(ds_train, validation_data=ds_test,
                epochs=30,
                steps_per_epoch = ((2000)//batch_size) , # * n_files_train,
                validation_steps = ((2000)//batch_size),  #* n_files_test,
                callbacks=[callback,csv] )


generator.generator_model.summary()

generator.save(model_save_location + model_name,save_format='tf')

try:
    training_stats = np.load(model_save_location + model_name + '/checkpoint/training_stats.npy', allow_pickle=True)
except:
    training_stats = []

# Will plot Loss (LLH) with number of data batches. Save these and data batch size

training_losses = []
validation_losses = []
data_batches = []
data_batches = []

for i in range(len(training_stats)):
    training_loss, validation_loss, batch, batch_size = training_stats[i]
    training_losses.append(training_loss)
    validation_losses.append(validation_loss)
    data_batches.append(i)
    data_batches.append(batch_size )
for i in history.history['loss']:
    training_losses.append(i)
for i in history.history['val_loss']:
    validation_losses.append(i)
print( training_losses)
for i in range(len(training_losses )):
    data_batches.append(i + len(training_stats))
    data_batches.append(batch_size )

format_plot(1)
plt.plot(training_losses, color='blue', label='Training')
plt.plot(validation_losses, color='magenta', label='Validation')
plt.title('Model Loss')
plt.ylabel('Unbinned Poisson LLH')
plt.xlabel('Data Batch')
plt.legend(loc='upper left')
plt.savefig(model_save_location + model_name + 'model_loss')

np.save(model_save_location + model_name + '/checkpoint/training_stats', [training_losses, validation_losses, data_batches, len(paths_train)] )
