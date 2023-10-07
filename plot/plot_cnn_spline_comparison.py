import numpy as np
from numpy import random
from math import sin, acos, cos, pi, asin, atan
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import sys
import math
import h5py


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

def arcdist(p1_zenith,p1_azimuth,p2_zenith,p2_azimuth):
        return 2*asin((sin((p1_zenith - p2_zenith)/2)**2 + cos(p1_zenith - pi/2)*cos(p2_zenith - pi/2)*sin((p1_azimuth - p2_azimuth)/2)**2)**0.5)
  
def zenithdist(p1_zenith,p1_azimuth,p2_zenith,p2_azimuth):
        p1_azimuth = np.pi/2.0
        p2_azimuth = np.pi/2.0
        return 2*asin((sin((p1_zenith - p2_zenith)/2)**2 + cos(p1_zenith - pi/2)*cos(p2_zenith - pi/2)*sin((p1_azimuth - p2_azimuth)/2)**2)**0.5)

def azimuthdist(p1_zenith,p1_azimuth,p2_zenith,p2_azimuth):
        p1_zenith = np.pi/2.0
        p2_zenith = np.pi/2.0
        return 2*asin((sin((p1_zenith - p2_zenith)/2)**2 + cos(p1_zenith - pi/2)*cos(p2_zenith - pi/2)*sin((p1_azimuth - p2_azimuth)/2)**2)**0.5)

  
def get_quartiles(dist):
    return [ np.percentile(dist, 25), np.percentile(dist, 50), np.percentile(dist, 75)  ]



modelname = sys.argv[1]


num_success_muongun_table = 0
num_success_muongun_cnn = 0
num_success_nue_table = 0
num_success_nue_cnn = 0

num_muongun = 250
num_nue = 250

num_muongun_total = num_muongun*5.0
num_nue_total = num_nue*25.0

#for both NuE and MuonGun:

#plot angular error dist of each per decade in energy (4)
#plot average (medians) angular error binned in energy (1)
#plot average (medians) angular error binned in num photons, with stat floor (1)
#plot energy reconstruction matrices (2)
#look at llh ratio to MC Truth histograms (1)
#plot average (medians) reco time binned in energy (1)

nue_table_mc_energy = []
nue_table_mc_charge = []
nue_table_mc_energy_contained = []
nue_table_mc_charge_contained = []


nue_cnn_mc_energy = []
nue_cnn_mc_charge = []
nue_cnn_mc_energy_contained  = []
nue_cnn_mc_charge_contained  = []

nue_table_reco_energy = []
nue_cnn_reco_energy = []
nue_table_reco_energy_contained  = []
nue_cnn_reco_energy_contained  = []

nue_table_angular_error = []
nue_cnn_angular_error = []
nue_table_angular_error_contained  = []
nue_cnn_angular_error_contained  = []

nue_table_mct_angular_error = []
nue_cnn_mct_angular_error = []
nue_table_mct_angular_error_contained  = []
nue_cnn_mct_angular_error_contained  = []

nue_table_zenith_error_contained  = []
nue_cnn_zenith_error_contained  = []
nue_table_azimuth_error_contained  = []
nue_cnn_azimuth_error_contained  = []

nue_table_mc_zenith_contained  = []
nue_cnn_mc_zenith_contained  = []
nue_table_mc_azimuth_contained  = []
nue_cnn_mc_azimuth_contained  = []


nue_table_time = []
nue_cnn_time = []

nue_table_time_contained = []
nue_cnn_time_contained = []


muongun_table_mc_energy = []
muongun_table_mc_charge = []

muongun_cnn_mc_energy = []
muongun_cnn_mc_charge = []

muongun_table_reco_energy = []
muongun_cnn_reco_energy = []

muongun_table_angular_error = []
muongun_cnn_angular_error = []

muongun_table_mct_angular_error = []
muongun_cnn_mct_angular_error = []

muongun_table_time = []
muongun_cnn_time = []

muongun_table_llh_ratio = []
muongun_cnn_llh_ratio = []

for i in range(num_nue):
    file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_NuE_' +  str(i)  + '.hdf5'
    try:
        f = h5py.File(file_str, 'r')
        for z in range( len(f['I3EventHeader'])):
            if f['Monopod_0photons_4iterations'][z]['fit_status'] == 0:
                vertex_x = f['Vertex_X'][z][5]
                vertex_y = f['Vertex_Y'][z][5]
                vertex_z = f['Vertex_Z'][z][5]
                
                #vertex_x = f['Monopod_0photons_4iterations'][z]['x']
                #vertex_y = f['Monopod_0photons_4iterations'][z]['y']
                #vertex_z = f['Monopod_0photons_4iterations'][z]['z']
                
                print( i)
                
                if (-450.0 < vertex_z) and (vertex_z < 450.0) and (np.sqrt(vertex_y**2 + vertex_x**2 ) < 450.0) and not (-200.0 < vertex_z and vertex_z < -50.0):
                
                    num_success_nue_table = num_success_nue_table + 1
                
                    nue_table_mc_energy_contained.append(f['MC_Truth'][z]['energy'])
                    nue_table_mc_charge_contained.append(f['Qtot'][z][5])
                    nue_table_reco_energy_contained.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                    nue_table_angular_error_contained.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                    nue_table_mct_angular_error_contained.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                    nue_table_time_contained.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                    
                    nue_table_mc_zenith_contained.append(f['MC_Truth'][z]['zenith'])
                    nue_table_mc_azimuth_contained.append(f['MC_Truth'][z]['azimuth'])
                    
                    nue_table_zenith_error_contained.append( (180.0/np.pi)*zenithdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                    
                    nue_table_azimuth_error_contained.append( (180.0/np.pi)*azimuthdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )



                    
                nue_table_mc_energy.append(f['MC_Truth'][z]['energy'])
                nue_table_mc_charge.append(f['Qtot'][z][5])
                nue_table_reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                nue_table_angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                nue_table_mct_angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                nue_table_time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                
        f.close()
    except Exception as ex:
        print(ex)
        f.close()

for i in range(num_nue):
    file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/VarAuto_Reco_NuE_' +  str(i) + '_' + modelname + '.hdf5'
    try:
        f = h5py.File(file_str, 'r')
        for z in range( len(f['I3EventHeader'])):
            if f['Monopod_0photons_4iterations'][z]['fit_status'] == 0:
                vertex_x = f['Vertex_X'][z][5]
                vertex_y = f['Vertex_Y'][z][5]
                vertex_z = f['Vertex_Z'][z][5]
                
                #vertex_x = f['Monopod_0photons_4iterations'][z]['x']
                #vertex_y = f['Monopod_0photons_4iterations'][z]['y']
                #vertex_z = f['Monopod_0photons_4iterations'][z]['z']

                
                if (-450.0 < vertex_z) and (vertex_z < 450.0) and (np.sqrt(vertex_y**2 + vertex_x**2 ) < 450.0) and not (-200.0 < vertex_z and vertex_z < -50.0):
                
                    num_success_nue_cnn = num_success_nue_cnn + 1
                
                    nue_cnn_mc_energy_contained.append(f['MC_Truth'][z]['energy'])
                    nue_cnn_mc_charge_contained.append(f['Qtot'][z][5])
                    nue_cnn_reco_energy_contained.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                    nue_cnn_angular_error_contained.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                    nue_cnn_mct_angular_error_contained.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                    nue_cnn_time_contained.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                    
                    nue_cnn_mc_zenith_contained.append(f['MC_Truth'][z]['zenith'])
                    nue_cnn_mc_azimuth_contained.append(f['MC_Truth'][z]['azimuth'])

                    nue_cnn_zenith_error_contained.append( (180.0/np.pi)*zenithdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )

                    nue_cnn_azimuth_error_contained.append( (180.0/np.pi)*azimuthdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )

                    
                nue_cnn_mc_energy.append(f['MC_Truth'][z]['energy'])
                nue_cnn_mc_charge.append(f['Qtot'][z][5])
                nue_cnn_reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                nue_cnn_angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                nue_cnn_mct_angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                nue_cnn_time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])

        f.close()
    except Exception as e:
        print(e)
        f.close()
        
        
print( len( nue_cnn_mc_energy))
print( len( nue_cnn_angular_error))

'''
for i in range(num_muongun):
    file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/VarAuto_Reco_MuonGun_' +  str(i) + '_' + modelname + '.hdf5'
    try:
        f = h5py.File(file_str, 'r')
        for z in range( len(f['Qtot'])):
            if f['FirstMillipedeFit_10mSpacing'][z]['fit_status'] == 0:
                num_success_muongun_cnn = num_success_muongun_cnn + 1

                event_id = f['Qtot'][z][1]
                
                mc_azimith = 0
                mc_zenith = 0
                mc_energy = 0

                for r in range( len(f['I3MCTree'] ) ):
                    if f['I3MCTree'][r][1] == event_id:
                        mc_azimith = f['I3MCTree'][r]['azimuth']
                        mc_zenith = f['I3MCTree'][r]['zenith']
                        mc_energy = f['I3MCTree'][r]['energy']
                        break

                #muongun_cnn_mc_energy.append(f['MC_Truth'][z]['energy'])
                muongun_cnn_mc_energy.append(mc_energy)
                muongun_cnn_mc_charge.append(f['Qtot'][z][5])
                muongun_cnn_reco_energy.append(f['FirstMillipedeFit_10mSpacing'][z]['energy'] )
                #muongun_cnn_angular_error.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                muongun_cnn_angular_error.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing'][z]['azimuth'],mc_zenith,mc_azimith)  )
                print(  (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing'][z]['azimuth'],mc_zenith,mc_azimith)  )
                #muongun_cnn_time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                #muongun_cnn_llh_ratio.append( -2.0*( f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['logl'] - f['FirstMillipedeFit_10mSpacing_MillipedeSeed'][z]['logl']) )
        f.close()
    except:
        f.close()
'''
'''
for i in range(num_muongun):
    file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_MuonGun_' +  str(i)  + '.hdf5'
    try:
        f = h5py.File(file_str, 'r')
        for z in range( len(f['Qtot'])):
            if f['SecondMillipedeFit_1mSpacing'][z]['fit_status'] == 0:
                num_success_muongun_table = num_success_muongun_table + 1
            
                muongun_table_mc_energy.append(f['MC_Truth'][z]['energy'])
                muongun_table_mc_charge.append(f['Qtot'][z][5])
                muongun_table_reco_energy.append(f['SecondMillipedeFit_1mSpacing'][z]['energy'] )
                muongun_table_angular_error.append( (180.0/np.pi)*arcdist(f['SecondMillipedeFit_1mSpacing'][z]['zenith'] ,f['SecondMillipedeFit_1mSpacing'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                muongun_table_time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                muongun_table_llh_ratio.append( -2.0*( f['SecondMillipedeFit_1mSpacing_MCTSeed'][z]['logl'] - f['SecondMillipedeFit_1mSpacing_MillipedeSeed'][z]['logl']) )
        f.close()
    except:
        f.close()
'''

print("NuE Table Reco Successes: ", num_success_nue_table/num_nue_total )
print("NuE CNN Reco Successes: ", num_success_nue_cnn/num_nue_total )
print("MuonGun Table Reco Successes: ", num_success_muongun_table/num_muongun_total )
print("MuonGun CNN Reco Successes: ", num_success_muongun_cnn/num_muongun_total )



def plot_angular_error_dist( table_mc_energy, cnn_mc_energy, table_angular_error, cnn_angular_error, title_str, save_str):

    table_4_5_angular_error = []
    table_5_6_angular_error = []
    table_6_7_angular_error = []
    table_7_8_angular_error = []
    
    cnn_4_5_angular_error = []
    cnn_5_6_angular_error = []
    cnn_6_7_angular_error = []
    cnn_7_8_angular_error = []
    
    for i in range( len(table_mc_energy )):
        if (1.0e4 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e5):
            table_4_5_angular_error.append( table_angular_error[i] )
        if (1.0e5 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e6):
            table_5_6_angular_error.append( table_angular_error[i] )
        if (1.0e6 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e7):
            table_6_7_angular_error.append( table_angular_error[i] )
        if (1.0e7 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e8):
            table_7_8_angular_error.append( table_angular_error[i] )

    for i in range( len(cnn_mc_energy )):
        if (1.0e4 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e5):
            cnn_4_5_angular_error.append( cnn_angular_error[i] )
        if (1.0e5 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e6):
            cnn_5_6_angular_error.append( cnn_angular_error[i] )
        if (1.0e6 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e7):
            cnn_6_7_angular_error.append( cnn_angular_error[i] )
        if (1.0e7 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e8):
            cnn_7_8_angular_error.append( cnn_angular_error[i] )
    
    bins = np.linspace(0.0, 180.0, 180 )
    
    ax =  format_plot(1)
    plt.hist(table_4_5_angular_error, bins, alpha=0.35, color='orange', edgecolor = 'orange', label='Table-based')
    plt.hist(cnn_4_5_angular_error, bins, alpha=0.35, color='cyan', edgecolor = 'cyan', label='CNN-based')
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e4 - 1.0e5 GeV)' )
    plt.vlines(np.median( table_4_5_angular_error), 0.0,40.0, linestyles ="dashed", colors ="orange")
    plt.vlines(np.median( cnn_4_5_angular_error), 0.0,40.0, linestyles ="dashed", colors ="cyan" )
    plt.vlines( -1.0, 0.0,40.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,40.0 )
    plt.legend( )
    
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_4_5_GeV', dpi=600)
    plt.close()
    
    ax =  format_plot(1)
    plt.hist(table_5_6_angular_error, bins, alpha=0.35, color='orange', edgecolor = 'orange', label='Table-based')
    plt.hist(cnn_5_6_angular_error, bins, alpha=0.35, color='cyan', edgecolor = 'cyan', label='CNN-based')
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e5 - 1.0e6 GeV)' )
    plt.vlines(np.median( table_5_6_angular_error), 0.0,55.0, linestyles ="dashed", colors ="orange")
    plt.vlines(np.median( cnn_5_6_angular_error), 0.0,55.0, linestyles ="dashed", colors ="cyan" )
    plt.vlines( -1.0, 0.0,55.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,55.0 )
    plt.legend( )
    
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_5_6_GeV', dpi=600)
    plt.close()
    
    ax =  format_plot(1)
    plt.hist(table_6_7_angular_error, bins, alpha=0.35, color='orange', edgecolor = 'orange', label='Table-based')
    plt.hist(cnn_6_7_angular_error, bins, alpha=0.35, color='cyan', edgecolor = 'cyan', label='CNN-based')
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e6 - 1.0e7 GeV)' )
    plt.vlines(np.median( table_6_7_angular_error), 0.0,60.0, linestyles ="dashed", colors ="orange")
    plt.vlines(np.median( cnn_6_7_angular_error), 0.0,60.0, linestyles ="dashed", colors ="cyan" )
    plt.vlines( -1.0, 0.0,60.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,60.0 )
    plt.legend( )

    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_6_7_GeV', dpi=600)
    plt.close()


def plot_single_angular_error_energy_curve( mc_energy, angular_error, title_str, save_str,energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    _25_quartiles = []
    _50_quartiles = []
    _75_quartiles = []
    
    for i in range(len(bins) - 1):
        angular_dist = []
        for j in range(len( mc_energy )):
            if (bins[i] < mc_energy[j]) and (mc_energy[j] <= bins[i+1]):
                angular_dist.append( angular_error[j] )
    
        _25_quartile, _50_quartile, _75_quartile = get_quartiles( angular_dist )

        _25_quartiles.append( _25_quartile)
        _50_quartiles.append( _50_quartile)
        _75_quartiles.append( _75_quartile)

    ax =  format_plot(1)
    plt.xlabel('Primary Energy (MC Truth) [GeV]')
    plt.ylabel('Angular Error [deg.]')
    plt.fill_between(bins_mids,_75_quartiles,_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,_50_quartiles, color='orange')
    plt.xscale('log')
    #plt.ylim(0.0, 180.0)
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()




def plot_angular_error_energy_curve( table_mc_energy, cnn_mc_energy, table_angular_error, cnn_angular_error, title_str, save_str, energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
    
    cnn_25_quartiles = []
    cnn_50_quartiles = []
    cnn_75_quartiles = []
    
    for i in range(len(bins) - 1):
        cnn_angular_dist = []
        table_angular_dist = []
        for j in range(len( table_mc_energy )):
            if (bins[i] < table_mc_energy[j]) and (table_mc_energy[j] <= bins[i+1]):
                table_angular_dist.append( table_angular_error[j] )
                
        for j in range(len( cnn_mc_energy )):
            if (bins[i] < cnn_mc_energy[j]) and (cnn_mc_energy[j] <= bins[i+1]):
                cnn_angular_dist.append( cnn_angular_error[j] )
                
    
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_angular_dist )

        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)
        
        cnn_25_quartiles.append( cnn_25_quartile)
        cnn_50_quartiles.append( cnn_50_quartile)
        cnn_75_quartiles.append( cnn_75_quartile)


    ax =  format_plot(1)
    plt.xlabel('Primary Energy (MC Truth) [GeV]')
    plt.ylabel('Angular Error [deg.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,table_50_quartiles, color='orange', label='Table-based Quartiles')
    plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
    plt.legend(loc='upper right' )
    plt.xscale('log')
    #plt.ylim(0.0, 180.0)
    plt.title(title_str )
    plt.xlim(bins_mids[0], bins_mids[-1] )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()

def plot_single_angular_error_charges_curve( mc_charge, angular_error, title_str, save_str, energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    _25_quartiles = []
    _50_quartiles = []
    _75_quartiles = []
    
    for i in range(len(bins) - 1):
        angular_dist = []
        for j in range(len( mc_charge )):
            if (bins[i] < mc_charge[j]) and (mc_charge[j] <= bins[i+1]):
                angular_dist.append( angular_error[j] )
                    
        _25_quartile, _50_quartile, _75_quartile = get_quartiles( angular_dist )
        
        _25_quartiles.append( _25_quartile)
        _50_quartiles.append( _50_quartile)
        _75_quartiles.append( _75_quartile)

    curve_points = np.logspace(3,7, 1000)
    curve_vals = []
    for i in range(len(curve_points)):
        curve_vals.append(np.sqrt( curve_points[i]) )


    ax =  format_plot(1)
    plt.xlabel('Deposited Charge [p.e.]')
    plt.ylabel('Angular Error [deg.]')
    plt.fill_between(bins_mids,_75_quartiles,_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,_50_quartiles, color='cyan')
    plt.xscale('log')
    #plt.ylim(0.0, 180.0)
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()


# nue_table_mc_charge, nue_cnn_mc_charge, nue_table_mct_angular_error, nue_cnn_mct_angular_error
def plot_angular_error_charges_curve( table_mc_charge, cnn_mc_charge, table_angular_error, cnn_angular_error, title_str, save_str, energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
    
    cnn_25_quartiles = []
    cnn_50_quartiles = []
    cnn_75_quartiles = []
    
    for i in range(len(bins) - 1):
        cnn_angular_dist = []
        table_angular_dist = []
        for j in range(len( table_mc_charge )):
            if (bins[i] < table_mc_charge[j]) and (table_mc_charge[j] <= bins[i+1]):
                table_angular_dist.append( table_angular_error[j] )
                
        for j in range(len( cnn_mc_charge )):
            if (bins[i] < cnn_mc_charge[j]) and (cnn_mc_charge[j] <= bins[i+1]):
                cnn_angular_dist.append( cnn_angular_error[j] )
                

        print(len( table_angular_dist), len(cnn_angular_dist ) , bins[i], bins[i+1] )
        #table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_angular_dist )
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        
        
        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)
        
        cnn_25_quartiles.append( cnn_25_quartile)
        cnn_50_quartiles.append( cnn_50_quartile)
        cnn_75_quartiles.append( cnn_75_quartile)

    curve_points = np.logspace(3,7, 1000)
    curve_vals = []
    for i in range(len(curve_points)):
        curve_vals.append(np.sqrt( curve_points[i]) )


    ax =  format_plot(1)
    plt.xlabel('Deposited Charge [p.e.]')
    plt.ylabel('Angular Error [deg.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,table_50_quartiles, color='orange', label='Table-based Quartiles')
    plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
    plt.legend(loc='upper right' )
    plt.xscale('log')
    #plt.ylim(0.0, 180.0)
    plt.title(title_str )
    plt.xlim(bins_mids[0], bins_mids[-1] )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()

def plot_zenith_resolution(table_zenith_error, cnn_zenith_error,event_zeniths,title_str, save_str  ):
    
    # Make containment bands as a function of zenith, azimuth
    bins = np.linspace(-1.0, 1.0, 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
    
    cnn_25_quartiles = []
    cnn_50_quartiles = []
    cnn_75_quartiles = []
    
    for i in range(len(bins) - 1):
        cnn_angular_dist = []
        table_angular_dist = []
        for j in range(len( event_zeniths )):
            if (bins[i] < np.cos(event_zeniths[j])) and (np.cos(event_zeniths[j]) <= bins[i+1]):
                table_angular_dist.append( table_zenith_error[j] )
                
        for j in range(len( event_zeniths )):
            if (bins[i] < np.cos(event_zeniths[j])) and (np.cos(event_zeniths[j]) <= bins[i+1]):
                cnn_angular_dist.append( cnn_zenith_error[j] )
                

        print(len( table_angular_dist), len(cnn_angular_dist ) , bins[i], bins[i+1] )
        #table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_angular_dist )
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        
        
        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)
        
        cnn_25_quartiles.append( cnn_25_quartile)
        cnn_50_quartiles.append( cnn_50_quartile)
        cnn_75_quartiles.append( cnn_75_quartile)

    curve_points = np.logspace(3,7, 1000)
    curve_vals = []
    for i in range(len(curve_points)):
        curve_vals.append(np.sqrt( curve_points[i]) )


    ax =  format_plot(1)
    plt.xlabel('Cos(True Zenith)')
    plt.ylabel('Zenith Error [deg.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,table_50_quartiles, color='orange', label='Table-based Quartiles')
    plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
    plt.legend(loc='upper right' )
    #plt.ylim(0.0, 180.0)
    plt.xlim(bins_mids[0], bins_mids[-1])
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()


def plot_azimuth_resolution(table_azimuth_error, cnn_azimuth_error, event_azimuths ,title_str, save_str  ):
    
    # Make containment bands as a function of zenith, azimuth
    bins = np.linspace(0, 2.0*np.pi, 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
    
    cnn_25_quartiles = []
    cnn_50_quartiles = []
    cnn_75_quartiles = []
    
    for i in range(len(bins) - 1):
        cnn_angular_dist = []
        table_angular_dist = []
        for j in range(len( event_azimuths )):
            if (bins[i] < event_azimuths[j]) and (event_azimuths[j] <= bins[i+1]):
                table_angular_dist.append( table_azimuth_error[j] )
                
        for j in range(len( event_azimuths )):
            if (bins[i] < event_azimuths[j]) and (event_azimuths[j] <= bins[i+1]):
                cnn_angular_dist.append( cnn_azimuth_error[j] )
                

        print(len( table_angular_dist), len(cnn_angular_dist ) , bins[i], bins[i+1] )
        #table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_angular_dist )
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        
        
        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)
        
        cnn_25_quartiles.append( cnn_25_quartile)
        cnn_50_quartiles.append( cnn_50_quartile)
        cnn_75_quartiles.append( cnn_75_quartile)

    curve_points = np.logspace(3,7, 1000)
    curve_vals = []
    for i in range(len(curve_points)):
        curve_vals.append(np.sqrt( curve_points[i]) )


    ax =  format_plot(1)
    plt.xlabel('True Azimuth [radians]')
    plt.ylabel('Azimuth Error [deg.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,table_50_quartiles, color='orange', label='Table-based Quartiles')
    plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
    plt.legend(loc='upper right' )
    plt.xlim(bins_mids[0], bins_mids[-1])
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()



def plot_angular_error_comparison( table_angular_error, cnn_angular_error, event_MC_energies, title_str, save_str):

    fig1 = format_plot(1)
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(table_angular_error, cnn_angular_error,c=np.log10(event_MC_energies), cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel('Deposited Charge [p.e.]', rotation=270)
    plt.xlabel('Table Angular Resolution [deg.]')
    plt.ylabel('CNN Angular Resolution [deg.]')
    plt.title(title_str)
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str, dpi=600)
    plt.close()
    

#nue_cnn_mc_energy,nue_cnn_reco_energy,1.0e4,1.0e7, 100, 'NuE - Monopod' ,'monopod_nue_table_energy_matrix_new'
def plot_energy_resolution(event_MC_energies,event_reco_energies,energy_range_low,energy_range_high, bins, title_str ,save_str):

    decade_counts = np.zeros( bins )
    energy_bins = np.logspace( np.log10(energy_range_low ),np.log10(energy_range_high ), bins + 1 )

    for i in range(len( event_MC_energies)):
        for j in range(len(decade_counts)):
            if energy_bins[j] <= event_MC_energies[i] and event_MC_energies[i] < energy_bins[j + 1]:
                decade_counts[j] = decade_counts[j] + 1.0

    counts,xedges, yedges = np.histogram2d(event_MC_energies, event_reco_energies,bins=[energy_bins, energy_bins] )

    for i in range(len(counts)):
        for j in range(len(counts[i])):
            counts[i][j] = counts[i][j]/1.0 #decade_counts[i]

    counts = np.ma.masked_where(counts == 0.0, counts)
    cmap = plt.cm.get_cmap('magma_r')
    cmap.set_bad(color='white')

    fig1 = format_plot(1)
    colors = plt.pcolormesh(energy_bins, energy_bins, counts,cmap=cmap)
    c = plt.colorbar(colors)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Reco. Energy [GeV]')
    plt.ylabel('Primary Energy (MC Truth) [GeV]')
    plt.xlim( energy_range_low,energy_range_high)
    plt.ylim( energy_range_low,energy_range_high)
    plt.title(title_str)
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str, dpi=600)
    plt.title(title_str)
    plt.close()


def plot_compared_resolution(table_angular_error, cnn_angular_error,event_MC_charges,energy_range_low,energy_range_high, degree_bins, title_str ,save_str):

    '''
    decade_counts = np.zeros(len( degree_bins) )

    for i in range(len( cnn_angular_error)):
        for j in range(len(decade_counts)):
            if degree_bins[j] <= cnn_angular_error[i] and cnn_angular_error[i] < degree_bins[j + 1]:
                if energy_range_low < event_MC_energies[i] and energy_range_high < event_MC_energies[i]:
                    decade_counts[j] = decade_counts[j] + 1.0
    '''
    
    cnn_angular_error_cut = []
    table_angular_error_cut = []
    
    for i in range(len(table_angular_error )):
        if (energy_range_low < event_MC_charges[i]) and (event_MC_charges[i] < energy_range_high):
            cnn_angular_error_cut.append( table_angular_error[i] )
            table_angular_error_cut.append( cnn_angular_error[i] )

    
    counts,xedges, yedges = np.histogram2d(cnn_angular_error_cut, table_angular_error_cut,bins=[degree_bins, degree_bins] )

    for i in range(len(counts)):
        for j in range(len(counts[i])):
            counts[i][j] = counts[i][j]/1.0 #decade_counts[i]

    counts = np.ma.masked_where(counts == 0.0, counts)
    cmap = plt.cm.get_cmap('magma_r')
    cmap.set_bad(color='white')

    fig1 = format_plot(1)
    colors = plt.pcolormesh(degree_bins, degree_bins, counts,cmap=cmap)
    c = plt.colorbar(colors)
    plt.xlabel('CNN Angular Resolution [deg.]')
    plt.ylabel('Table Angular Resolution [deg.]')
    plt.xlim( degree_bins[0], degree_bins[-1])
    plt.ylim( degree_bins[0], degree_bins[-1])
    plt.title(title_str)
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str, dpi=600)
    plt.title(title_str)
    plt.close()


def plot_llh_histos(table_llh_ratio,cnn_llh_ratio, title_str ,save_string):
    
    bins = np.linspace(0.0, 500, 50)
    
    ax =  format_plot(1)
    plt.xlabel(r'$-2 \Delta LLH$' )
    plt.hist(table_llh_ratio, bins=bins, alpha=0.4, color='orange', edgecolor = 'orange', label='Table-based' )
    plt.hist(cnn_llh_ratio, bins=bins, alpha=0.4, color='cyan', edgecolor = 'cyan', label='CNN-based' )
    plt.legend(loc='upper right' )
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()


def plot_runtime_energy_curve( table_mc_energy, cnn_mc_energy, table_time, cnn_time, title_str, save_str, energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
    
    cnn_25_quartiles = []
    cnn_50_quartiles = []
    cnn_75_quartiles = []
    
    for i in range(len(bins) - 1):
        cnn_time_dist = []
        table_time_dist = []
        for j in range(len( table_mc_energy )):
            if (bins[i] < table_mc_energy[j]) and (table_mc_energy[j] <= bins[i+1]):
                table_time_dist.append( table_time[j] )
                
        for j in range(len( cnn_mc_energy )):
            if (bins[i] < cnn_mc_energy[j]) and (cnn_mc_energy[j] <= bins[i+1]):
                cnn_time_dist.append( cnn_time[j] )
                
    
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_time_dist )
        cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_time_dist )

        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)
        
        cnn_25_quartiles.append( cnn_25_quartile)
        cnn_50_quartiles.append( cnn_50_quartile)
        cnn_75_quartiles.append( cnn_75_quartile)


    ax =  format_plot(1)
    plt.xlabel('Primary Energy (MC Truth) [GeV]')
    plt.ylabel('Runtime [sec.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='orange', alpha=0.4)
    plt.plot(bins_mids,table_50_quartiles, color='orange', label='Table-based Quartiles')
    plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, color='cyan', alpha=0.4)
    plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
    plt.legend(loc='upper right' )
    plt.xscale('log')
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()



# nue_table_mc_energy_contained
# nue_table_mc_charge_contained
# nue_cnn_mc_energy_contained
# nue_cnn_mc_charge_contained

# Both sets of events correspond to the same original set, same charge
plot_compared_resolution(nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained,-np.inf,100.0, np.linspace(0.0, 40.0, 15),'(Edep < 1.0E2 p.e.)' ,'monopod_nue_angular_error_comparison_1e2_contained_' + modelname)
plot_compared_resolution(nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained,100.0,1.0e3, np.linspace(0.0, 40.0, 15),'(1.0E2 p.e. < Edep < 1.0E3 p.e.)' ,'monopod_nue_angular_error_comparison_1e3_contained_' + modelname)
plot_compared_resolution(nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained,1.0e3,1.0e4, np.linspace(0.0, 40.0, 15),'(1.0E3 p.e. < Edep < 1.0E4 p.e.)' ,'monopod_nue_angular_error_comparison_1e4_contained_' + modelname)
plot_compared_resolution(nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained,1.0e4,1.0e5, np.linspace(0.0, 40.0, 15),'(1.0E4 p.e. < Edep < 1.0E5 p.e.)' ,'monopod_nue_angular_error_comparison_1e5_contained_' + modelname)
plot_compared_resolution(nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained,1.0e5,np.inf, np.linspace(0.0, 40.0, 15),'(1.0E5 p.e. < Edep)' ,'monopod_nue_angular_error_comparison_1e6_contained_' + modelname)


#plot_angular_error_dist( muongun_table_mc_energy, muongun_cnn_mc_energy, muongun_table_angular_error, muongun_cnn_angular_error, 'MuonGun - Millipede', 'millipede_muongun_angular_error_dist_' + modelname, 1.0e4, 1.0e8)
#plot_angular_error_energy_curve( muongun_table_mc_energy, muongun_cnn_mc_energy, muongun_table_angular_error, muongun_cnn_angular_error, 'MuonGun - Millipede', 'millipede_muongun_angular_error_energy_curve_' + modelname, 1.0e4, 1.0e8)
#plot_angular_error_charges_curve( muongun_table_mc_charge, muongun_cnn_mc_charge, muongun_table_angular_error, muongun_cnn_angular_error, 'MuonGun - Millipede', 'millipede_muongun_angular_error_charge_curve_' + modelname, 1.0e4, 1.0e8)
##plot_energy_resolution(muongun_table_mc_energy,muongun_table_reco_energy,1.0e4,1.0e8, 100, 'MuonGun - Millipede' ,'millipede_muongun_table_energy_matrix_' + modelname)
##plot_energy_resolution(muongun_cnn_mc_energy,muongun_cnn_reco_energy,1.0e4,1.0e8, 100, 'MuonGun - Millipede' ,'millipede_muongun_table_energy_matrix_' + modelname)
#plot_llh_histos(muongun_table_llh_ratio,muongun_cnn_llh_ratio, 'MuonGun - Millipede' , 'millipede_muongun_llh_dist_' + modelname)
#plot_runtime_energy_curve( muongun_table_mc_energy, muongun_cnn_mc_energy, muongun_table_time, muongun_cnn_time, 'MuonGun - Millipede', 'millipede_muongun_runtime_energy_curve_' + modelname, 1.0e4, 1.0e8)

plot_angular_error_comparison( nue_table_angular_error_contained, nue_cnn_angular_error_contained, nue_table_mc_charge_contained, 'NuE - Monopod, Contained', 'monopod_nue_angular_error_comparison_contained_' + modelname)

plot_azimuth_resolution(nue_table_azimuth_error_contained, nue_cnn_azimuth_error_contained, nue_table_mc_azimuth_contained,'NuE - Monopod, Contained', 'azimuth_resolution_contained_' + modelname)

plot_zenith_resolution(nue_table_zenith_error_contained, nue_cnn_zenith_error_contained, nue_table_mc_zenith_contained,'NuE - Monopod, Contained', 'zenith_resolution_contained_' + modelname)


plot_angular_error_dist( nue_table_mc_energy_contained, nue_cnn_mc_energy_contained, nue_table_angular_error_contained, nue_cnn_angular_error_contained, 'NuE - Monopod, Contained', 'monopod_nue_angular_error_dist_contained_' + modelname)
plot_angular_error_energy_curve( nue_table_mc_energy_contained, nue_cnn_mc_energy_contained, nue_table_angular_error_contained, nue_cnn_angular_error_contained, 'NuE - Monopod, Contained Events', 'monopod_nue_angular_error_energy_curve_contained_', 1.0e4, 1.0e7)
plot_angular_error_energy_curve( nue_table_mc_energy, nue_cnn_mc_energy, nue_table_angular_error, nue_cnn_angular_error, 'NuE - Monopod', 'monopod_nue_angular_error_energy_curve' + modelname, 1.0e4, 1.0e7)

plot_angular_error_charges_curve( nue_table_mc_charge_contained, nue_cnn_mc_charge_contained, nue_table_angular_error_contained, nue_cnn_angular_error_contained, 'NuE - Monopod', 'monopod_nue_angular_error_charge_curve_contained_' + modelname, 1.0e3, 1.0e6)
plot_angular_error_energy_curve( nue_table_mc_energy_contained, nue_cnn_mc_energy_contained,nue_table_angular_error_contained, nue_cnn_angular_error_contained, 'NuE - Monopod', 'monopod_nue_angular_error_energy_curve_contained_' + modelname, 1.0e4, 1.0e7)


###plot_angular_error_charges_curve( nue_table_mc_charge, nue_cnn_mc_charge, nue_table_mct_angular_error, nue_cnn_mct_angular_error, 'NuE - Monopod MC Truth Seeded', 'monopod_nue_angular_error_charge_curve_mct_' + modelname, 1.0e3, 1.0e6)
###plot_angular_error_charges_curve( nue_table_mc_charge, nue_cnn_mc_charge, nue_table_angular_error, nue_cnn_angular_error, 'NuE - Monopod', 'monopod_nue_angular_error_charge_curve_check_' + modelname, 1.0e3, 1.0e6)

plot_energy_resolution(nue_table_mc_energy,nue_table_reco_energy,1.0e4,1.0e7, 100, 'NuE - Monopod, Table-Based' ,'monopod_nue_table_energy_matrix_' + modelname)
plot_energy_resolution(nue_cnn_mc_energy,nue_cnn_reco_energy,1.0e4,1.0e7, 100, 'NuE - Monopod, CNN-Based' ,'monopod_nue_cnn_energy_matrix_' + modelname)
plot_energy_resolution(nue_table_mc_energy_contained,nue_table_reco_energy_contained,1.0e4,1.0e7, 100, 'NuE - Monopod, Table-Based' ,'monopod_nue_table_energy_matrix_contained_' + modelname)
plot_energy_resolution(nue_cnn_mc_energy_contained,nue_cnn_reco_energy_contained,1.0e4,1.0e7, 100, 'NuE - Monopod, CNN-Based' ,'monopod_nue_cnn_energy_matrix_contained_' + modelname)
#plot_runtime_energy_curve( nue_table_mc_energy, nue_cnn_mc_energy, nue_table_time, nue_cnn_time, 'NuE - Monopod', 'monopod_nue_runtime_energy_curve_' + modelname, 1.0e4, 1.0e7)

#plot_single_angular_error_energy_curve(  nue_cnn_mc_energy, nue_cnn_angular_error, 'NuE - Monopod', 'monopod_nue_angular_error_energy_curve_cnn_' + modelname, 1.0e4, 1.0e7)
#plot_single_angular_error_charges_curve(  nue_cnn_mc_charge, nue_cnn_angular_error, 'NuE - Monopod', 'monopod_nue_angular_error_charge_curve_cnn_' + modelname, 1.0e3, 1.0e6)

#plot_single_angular_error_charges_curve(  muongun_cnn_mc_charge, muongun_cnn_angular_error, 'MuonGun - Millipede, CNN-Based', 'millipede_muongun_angular_error_charge_curve_cnn_' + modelname, 1.0e3, 1.0e6)

