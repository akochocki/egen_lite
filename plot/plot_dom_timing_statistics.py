import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os,sys
from os.path import expandvars


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



event_zenith = []
event_azimith = []
event_z = []
event_r = []
event_total_charge = []
event_total_pred_charge = []
event_total_difference = []
event_total_timing_difference = []


per_event_charges = []
per_event_pred_charges = []
per_event_ratios = []
per_event_difs = []
per_event_timing_difs = []
per_event_z = []
per_event_x = []
per_event_y = []

all_dom_timing_diffs = []


quality_event_zenith = []
quality_event_azimith = []
quality_event_z = []
quality_event_r = []
quality_event_total_charge = []
quality_event_total_pred_charge = []
quality_event_total_difference = []
quality_event_total_timing_difference = []


quality_per_event_charges = []
quality_per_event_pred_charges = []
quality_per_event_ratios = []
quality_per_event_difs = []
quality_per_event_timing_difs = []
quality_per_event_z = []
quality_per_event_x = []
quality_per_event_y = []

quality_all_dom_timing_diffs = []
quality_all_dom_charges = []

tmin = -500.0
tmax = 4500.0

time_bins = np.linspace(tmin, tmax, num=250)
time_bins_mids = []
for k in range(len( time_bins) - 1):
    time_bins_mids.append((time_bins[k] + time_bins[k+1])/2.0 )

quality_pred_waveforms = []
quality_true_waveforms = []

model_str = sys.argv[1]

for i in range(10):
    f = np.load( '/mnt/scratch/kochocki/ftp_electrons/set_1/dom_time_stats_EMinus_' + str(1) + '_' + str(i) + '_' + model_str + '.npy', allow_pickle=True)
    for j in range(len(f)):
        real_zen, real_az, real_x, real_y, real_z, event_pulse_info = f[j]
        
        if (-450.0 < real_z) and (real_z < 450.0) and (np.sqrt(real_y**2 + real_x**2 ) < 450.0) and not (-200.0 < real_z and real_z < -50.0):

            quality_event_zenith.append( real_zen)
            quality_event_azimith.append( real_az)
            quality_event_z.append( real_z)
            quality_event_r.append( np.sqrt( (real_x)**2 + (real_y)**2  ) )
            
            total_true_charge = 0.0
            total_pred_charge = 0.0
            total_abs_dif = 0.0
            
            om_timing_diffs = []
            om_charges = []
            om_pred_charges = []
            om_ratios = []
            om_difs = []
            om_z = []
            om_x = []
            om_y = []
            om_pred_binned_charges = []
            om_true_binned_charges = []
            
            
            for k in range(len( event_pulse_info)):
                string, om, pulse_x, pulse_y, pulse_z, true_om_charge, pred_om_charge, om_charge_dif, om_charge_ratio, waveform_abs_dif, real_binned_charges, pred_binned_charges  = event_pulse_info[k]
                
                total_true_charge = total_true_charge + true_om_charge
                total_pred_charge = total_pred_charge + pred_om_charge[0]
                total_abs_dif = total_abs_dif + waveform_abs_dif
                
                om_charges.append( true_om_charge)
                om_pred_charges.append( pred_om_charge[0])
                om_ratios.append( om_charge_ratio[0])
                om_difs.append( np.abs(om_charge_dif[0]))
                om_z.append( pulse_z)
                om_x.append( pulse_x)
                om_y.append( pulse_y)
                om_timing_diffs.append( waveform_abs_dif)
                om_pred_binned_charges.append(  pred_binned_charges)
                om_true_binned_charges.append( real_binned_charges)
       
                
                quality_all_dom_timing_diffs.append(waveform_abs_dif ) # every difference per dom from all events
                quality_all_dom_charges.append( true_om_charge)
            
            quality_per_event_charges.append( om_charges)
            quality_per_event_pred_charges.append( om_pred_charges)
            quality_per_event_ratios.append( om_ratios)
            quality_per_event_difs.append( om_difs)
            quality_per_event_z.append( om_z)
            quality_per_event_x.append( om_x)
            quality_per_event_y.append( om_y)
            quality_per_event_timing_difs.append( om_timing_diffs)  # array per event of differences on doms
            
            quality_event_total_charge.append( total_true_charge )
            quality_event_total_pred_charge.append( total_pred_charge)
            quality_event_total_difference.append( np.abs(total_true_charge - total_pred_charge) )
            quality_event_total_timing_difference.append( total_abs_dif) # whole difference of each event array
            
            quality_pred_waveforms.append( om_pred_binned_charges)
            quality_true_waveforms.append( om_true_binned_charges)
            
        ##########
        
        event_zenith.append( real_zen)
        event_azimith.append( real_az)
        event_z.append( real_z)
        event_r.append( np.sqrt( (real_x)**2 + (real_y)**2  ) )
        
        total_true_charge = 0.0
        total_pred_charge = 0.0
        total_abs_dif = 0.0
        
        om_timing_diffs = []
        om_charges = []
        om_pred_charges = []
        om_ratios = []
        om_difs = []
        om_z = []
        om_x = []
        om_y = []
        
        for k in range(len( event_pulse_info)):
            string, om, pulse_x, pulse_y, pulse_z, true_om_charge, pred_om_charge, om_charge_dif, om_charge_ratio, waveform_abs_dif, real_binned_charges, pred_binned_charges  = event_pulse_info[k]
            
            total_true_charge = total_true_charge + true_om_charge
            total_pred_charge = total_pred_charge + pred_om_charge[0]
            total_abs_dif = total_abs_dif + waveform_abs_dif
            
            om_charges.append( true_om_charge)
            om_pred_charges.append( pred_om_charge[0])
            om_ratios.append( om_charge_ratio[0])
            om_difs.append( np.abs(om_charge_dif[0]))
            om_z.append( pulse_z)
            om_x.append( pulse_x)
            om_y.append( pulse_y)
            om_timing_diffs.append( waveform_abs_dif)
            
            all_dom_timing_diffs.append(waveform_abs_dif ) # every difference per dom from all events
        
        per_event_charges.append( om_charges)
        per_event_pred_charges.append( om_pred_charges)
        per_event_ratios.append( om_ratios)
        per_event_difs.append( om_difs)
        per_event_z.append( om_z)
        per_event_x.append( om_x)
        per_event_y.append( om_y)
        per_event_timing_difs.append( om_timing_diffs) # array per event of differences on doms
        
        event_total_charge.append( total_true_charge )
        event_total_pred_charge.append( total_pred_charge)
        event_total_difference.append( np.abs(total_true_charge - total_pred_charge) )
        event_total_timing_difference.append( total_abs_dif) # whole difference of each event array


# make histogram with event absolute value differences
# for events with largest absolute value differences (apply some cut), look at
# waveforms with largest differences
# difference per DOM as a function of true charge

event_absolute_value_cut = 0.0
waveform_absolute_value_cut = 75.0

print(quality_event_total_timing_difference)
print(quality_event_total_pred_charge)

sub_folder_name = '/mnt/home/kochocki/egen_lite/plots/event_examples'
if not os.path.exists(sub_folder_name ):
    os.mkdir(sub_folder_name )

format_plot(1)
plt.yscale('log')
plt.hist( quality_event_total_timing_difference, range=[0.0,np.amax(quality_event_total_timing_difference) ], bins=30, color='blue', alpha=0.3)
plt.xlim(0.0,np.amax(quality_event_total_timing_difference) )
plt.xlabel('Event Absolute Charge (Timed) Difference [p.e.]')
plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_event_charge_timed_dif' + model_str, dpi=300)
plt.close( )

format_plot(1)
plt.hist( quality_all_dom_timing_diffs, range=[0.0, np.amax( quality_all_dom_timing_diffs)], bins=30, color='blue', alpha=0.3)
plt.yscale('log')
plt.xlim( 0.0, np.amax(quality_all_dom_timing_diffs))
plt.xlabel('Per DOM Absolute Charge (Timed) Difference [p.e.]')
plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_dom_charge_timed_dif' + model_str, dpi=300)
plt.close( )


format_plot(1)
plt.scatter( quality_event_total_charge, quality_event_total_timing_difference, color='blue', alpha=0.1)
plt.xlim(0.0, np.amax(event_total_charge))
plt.ylim(-5.0,np.amax(quality_event_total_timing_difference) )
plt.plot( np.linspace(0.0, 1000.0, 2000), np.full(2000, 0.0 ), color='gray',alpha=0.3, linestyle='--'  )
plt.xlabel('Event Total Charge [p.e.]')
plt.ylabel('Event Absolute Charge (Timed) Difference [p.e.]')
plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_dom_charge_timed_dif_w_event' + model_str, dpi=300)
plt.close( )

format_plot(1)
plt.scatter( quality_all_dom_charges, quality_all_dom_timing_diffs, color='blue', alpha=0.1)
plt.xlim(0.0, 850.0)
plt.ylim(-5.0,400.0 )
plt.plot( np.linspace(0.0, 1000.0, 2000), np.full(2000, 0.0 ), color='gray',alpha=0.3, linestyle='--'  )
plt.plot( np.linspace(0.0, 1000.0, 2000), np.sqrt(np.linspace(0.0, 1000.0, 2000)), color='red',alpha=0.6 )
plt.xlabel('DOM Total Charge [p.e.]')
#plt.yscale('log')
#plt.xscale('log')
plt.ylabel('Per DOM Absolute Charge (Timed) Difference [p.e.]')
plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_dom_charge_timed_dif_w_dom' + model_str, dpi=300)
plt.close( )



for i in range(len(quality_per_event_charges)):


    # plot abs timing charge per dom distribution
    # plot abs timing charge per dom charge
    
    #print(i)
    
    if quality_event_total_timing_difference[i] > event_absolute_value_cut:

        format_plot(i)
        plt.hist( quality_per_event_timing_difs[i],range=[0.0, np.amax(quality_per_event_timing_difs[i])], bins=30, color='purple', alpha=0.3)
        plt.yscale('log')
        plt.xlim( 0.0, np.amax(quality_per_event_timing_difs[i]) )
        plt.xlabel('Per DOM Absolute Charge (Timed) Difference [p.e.]')
        plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_dom_charge_timed_dif' + str(i)  + model_str, dpi=300)
        plt.close( )

        format_plot(i)
        plt.scatter( quality_per_event_charges[i], quality_per_event_timing_difs[i], color='purple', alpha=0.1)
        plt.xlim(0.0, np.amax(quality_per_event_charges[i]) )
        plt.ylim(0.0, np.amax(quality_per_event_timing_difs[i] ) )
        plt.plot( np.linspace(0.0, 1000.0, 2000), np.full(2000, 0.0 ), color='gray',alpha=0.3, linestyle='--'  )
        plt.plot( np.linspace(0.0, 1000.0, 2000), np.sqrt(np.linspace(0.0, 1000.0, 2000)), color='red',alpha=0.6, linestyle='--'  )
        plt.ylabel('Per DOM Absolute Charge (Timed) Difference [p.e.]')
        plt.xlabel('True DOM Dep. Charge [p.e.]')
        plt.savefig( '/mnt/home/kochocki/egen_lite/plots/event_examples/quality_event_charge_dif_w_timed_dif' + str(i) + model_str, dpi=300)
        plt.close( )

        for j in range( len(quality_per_event_charges[i])):
        
            if quality_per_event_timing_difs[i][j] > waveform_absolute_value_cut:
            
                print(i, j)
                print(quality_true_waveforms[i][j] )
            
                fig = format_plot(1)
                plt.hist(time_bins_mids, time_bins, weights=quality_true_waveforms[i][j], range=[0., np.amax(quality_true_waveforms[i][j] + quality_pred_waveforms[i][j]) ], histtype=u'step', alpha=0.5,  color='magenta',linestyle='-', label=r'Real Binned Pulses')
                plt.hist(time_bins_mids, time_bins, weights=quality_pred_waveforms[i][j], range=[0., np.amax(quality_true_waveforms[i][j] + quality_pred_waveforms[i][j]) ], histtype=u'step', alpha=0.5, color='green',linestyle='-', label=r'Predicted Binned Pulses')
                plt.xlim( time_bins_mids[0], time_bins_mids[-1])
                plt.legend(loc="upper right")
                plt.savefig('/mnt/home/kochocki/egen_lite/plots/event_examples/waveform_' + str(i) + '_' + str(j) + model_str, dpi=200)
                plt.close()

