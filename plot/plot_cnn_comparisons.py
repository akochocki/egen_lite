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


modelnames = ['loss_model_ftp_v2_1_3_25_500']


# for models in the list of model names:
# plot times for reconstruction
# plot reconstruction angular error 68% containment as a function of energy
# plot angular error distribution binned

# for each plot, just reload information per model

def get_nue_info_cnn(model_name, contained ):

    num_success_nue = 0
    num_nue = 250
    num_nue_total = num_nue*25.0
    
    mc_energy = []
    mc_charge = []
    reco_energy = []
    angular_error = []
    angular_error_mct_seed = []
    time = []
    
    for i in range(num_nue):
        file_str = '/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/VarAuto_Reco_NuE_' +  str(i) + '_' + modelname + '.hdf5'
        try:
            f = h5py.File(file_str, 'r')
            for z in range( len(f['I3EventHeader'])):
                if f['Monopod_0photons_4iterations'][z]['fit_status'] == 0:
                    vertex_x = f['Vertex_X'][z][5]
                    vertex_y = f['Vertex_Y'][z][5]
                    vertex_z = f['Vertex_Z'][z][5]
                    
                    if contained:
                        if (-450.0 < vertex_z) and (vertex_z < 450.0) and (np.sqrt(vertex_y**2 + vertex_x**2 ) < 450.0) and not (-200.0 < vertex_z and vertex_z < -50.0):
                        
                            num_success_nue = num_success_nue + 1
                        
                            mc_energy.append(f['MC_Truth'][z]['energy'])
                            mc_charge.append(f['Qtot'][z][5])
                            reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                            angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                            angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                            time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                            
                    else:
                        mc_energy.append(f['MC_Truth'][z]['energy'])
                        mc_charge.append(f['Qtot'][z][5])
                        reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                        angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                        angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                        time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])

            f.close()
        except Exception as e:
            print(e)
            f.close()

    print("NuE Table Reco Successes: ", num_success_nue/num_nue_total )
    return [mc_energy, mc_charge, reco_energy, angular_error, angular_error_mct_seed, time]
    
def get_millipede_info_cnn(model_name):

    num_success_muongun = 0
    num_muongun = 250
    num_muongun_total = num_muongun*5.0
    
    mc_energy = []
    mc_charge = []
    reco_energy = []
    angular_error = []
    angular_error_mct_seed = []
    time = []
    #llh_ratio = []
    
    for i in range(num_muongun):
        file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/VarAuto_Reco_MuonGun_' +  str(i) + '_' + modelname + '.hdf5'
        try:
            f = h5py.File(file_str, 'r')
            for z in range( len(f['Qtot'])):
                if f['FirstMillipedeFit_10mSpacing'][z]['fit_status'] == 0:
                    num_success_muongun = num_success_muongun + 1

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


                    mc_energy.append(mc_energy)
                    mc_charge.append(f['Qtot'][z][5])
                    reco_energy.append(f['FirstMillipedeFit_10mSpacing'][z]['energy'] )
                    angular_error.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing'][z]['azimuth'],mc_zenith,mc_azimith)  )
                    angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['azimuth'],mc_zenith,mc_azimith)  )
                    time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                    llh_ratio.append( -2.0*( f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['logl'] - f['FirstMillipedeFit_10mSpacing_MillipedeSeed'][z]['logl']) )
            f.close()
        except:
            f.close()

    print("NuE CNN Reco Successes: ", num_success_nue/num_nue_total )
    return [mc_energy, mc_charge, reco_energy, angular_error, angular_error_mct_seed, time]


def get_nue_info_table( contained):

    num_success_nue = 0
    num_nue = 250
    num_nue_total = num_nue*25.0
    
    mc_energy = []
    mc_charge = []
    reco_energy = []
    angular_error = []
    angular_error_mct_seed = []
    time = []
    
    for i in range(num_nue):
        file_str = '/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_NuE_' +  str(i) + '.hdf5'
        try:
            f = h5py.File(file_str, 'r')
            for z in range( len(f['I3EventHeader'])):
                if f['Monopod_0photons_4iterations'][z]['fit_status'] == 0:
                    vertex_x = f['Vertex_X'][z][5]
                    vertex_y = f['Vertex_Y'][z][5]
                    vertex_z = f['Vertex_Z'][z][5]
                    
                    if contained:
                        if (-450.0 < vertex_z) and (vertex_z < 450.0) and (np.sqrt(vertex_y**2 + vertex_x**2 ) < 450.0) and not (-200.0 < vertex_z and vertex_z < -50.0):
                        
                            num_success_nue = num_success_nue + 1
                        
                            mc_energy.append(f['MC_Truth'][z]['energy'])
                            mc_charge.append(f['Qtot'][z][5])
                            reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                            angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                            angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                            time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                            
                    else:
                        mc_energy.append(f['MC_Truth'][z]['energy'])
                        mc_charge.append(f['Qtot'][z][5])
                        reco_energy.append(f['Monopod_0photons_4iterations'][z]['energy'] )
                        angular_error.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations'][z]['zenith'] ,f['Monopod_0photons_4iterations'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                        angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['Monopod_0photons_4iterations_MCTSeed'][z]['zenith'] ,f['Monopod_0photons_4iterations_MCTSeed'][z]['azimuth'],f['MC_Truth'][z]['zenith'],f['MC_Truth'][z]['azimuth'])  )
                        time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])

            f.close()
        except Exception as e:
            print(e)
            f.close()

    print("MuonGun CNN Reco Successes: ", num_success_muongun/num_muongun_total )
    return [mc_energy, mc_charge, reco_energy, angular_error, angular_error_mct_seed, time]
  
def get_millipede_info_table():

    num_success_muongun = 0
    num_muongun = 250
    num_muongun_total = num_muongun*5.0
    
    mc_energy = []
    mc_charge = []
    reco_energy = []
    angular_error = []
    angular_error_mct_seed = []
    time = []
    #llh_ratio = []
    
    for i in range(num_muongun):
        file_str = '/mnt/gs21/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_MuonGun_' +  str(i) + '_' + modelname + '.hdf5'
        try:
            f = h5py.File(file_str, 'r')
            for z in range( len(f['Qtot'])):
                if f['FirstMillipedeFit_10mSpacing'][z]['fit_status'] == 0:
                    num_success_muongun = num_success_muongun + 1

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


                    mc_energy.append(mc_energy)
                    mc_charge.append(f['Qtot'][z][5])
                    reco_energy.append(f['FirstMillipedeFit_10mSpacing'][z]['energy'] )
                    angular_error.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing'][z]['azimuth'],mc_zenith,mc_azimith)  )
                    angular_error_mct_seed.append( (180.0/np.pi)*arcdist(f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['zenith'] ,f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['azimuth'],mc_zenith,mc_azimith)  )
                    time.append(f['End_Reco_Time'][z][5] - f['Start_Reco_Time'][z][5])
                    llh_ratio.append( -2.0*( f['FirstMillipedeFit_10mSpacing_MCTSeed'][z]['logl'] - f['FirstMillipedeFit_10mSpacing_MillipedeSeed'][z]['logl']) )
            f.close()
        except:
            f.close()

    print("MuonGun Table Reco Successes: ", num_success_muongun/num_muongun_total )
    return [mc_energy, mc_charge, reco_energy, angular_error, angular_error_mct_seed, time]


def plot_angular_error_dist( models, type, title_str, save_str):

    if type == 'track':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_millipede_info_table()

    if type == 'cascade':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_nue_info_table(True )

    table_4_5_angular_error = []
    table_5_6_angular_error = []
    table_6_7_angular_error = []
    table_7_8_angular_error = []
    
    for i in range( len(table_mc_energy )):
        if (1.0e4 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e5):
            table_4_5_angular_error.append( table_angular_error[i] )
        if (1.0e5 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e6):
            table_5_6_angular_error.append( table_angular_error[i] )
        if (1.0e6 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e7):
            table_6_7_angular_error.append( table_angular_error[i] )
        if (1.0e7 < table_mc_energy[i]) and (table_mc_energy[i] <= 1.0e8):
            table_7_8_angular_error.append( table_angular_error[i] )
        
    cnn_4_5_angular_error = []
    cnn_5_6_angular_error = []
    cnn_6_7_angular_error = []
    cnn_7_8_angular_error = []
    

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
    
    table_mc_energy, cnn_mc_energy, table_angular_error, cnn_angular_error
    ax =  format_plot(1)
    plt.hist(table_4_5_angular_error, bins, alpha=0.8, color='black',histtype='step', edgecolor = 'black', label='Table-based')
    
    for j in range(len(models)):
        cnn_4_5_angular_error = []
        if type == 'track':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_millipede_info_cnn( models[j])

        if type == 'cascade':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_nue_info_cnn(models[j], True )

        for i in range( len(cnn_mc_energy )):
            if (1.0e4 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e5):
                cnn_4_5_angular_error.append( cnn_angular_error[i] )
        
        plt.hist(cnn_4_5_angular_error, bins, alpha=0.35,histtype='step', label='CNN-based model ' + str(j) )
        
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e4 - 1.0e5 GeV)' )
    #plt.vlines(np.median( table_4_5_angular_error), 0.0,40.0, linestyles ="dashed", colors ="orange")
    #plt.vlines(np.median( cnn_4_5_angular_error), 0.0,40.0, linestyles ="dashed", colors ="cyan" )
    #plt.vlines( -1.0, 0.0,40.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,40.0 )
    plt.legend( )
    
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_models_4_5_GeV', dpi=600)
    plt.close()
    
    
    ax =  format_plot(1)
    plt.hist(table_5_6_angular_error, bins, alpha=0.8, color='black',histtype='step', edgecolor = 'black', label='Table-based')
    for j in range(len(models)):
        cnn_5_6_angular_error = []
        if type == 'track':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_millipede_info_cnn( models[j])

        if type == 'cascade':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_nue_info_cnn(models[j], True )

        for i in range( len(cnn_mc_energy )):
            if (1.0e5 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e6):
                cnn_5_6_angular_error.append( cnn_angular_error[i] )
        
        plt.hist(cnn_5_6_angular_error, bins, alpha=0.35,histtype='step', label='CNN-based')
        
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e5 - 1.0e6 GeV)' )
    #plt.vlines(np.median( table_5_6_angular_error), 0.0,55.0, linestyles ="dashed", colors ="orange")
    #plt.vlines(np.median( cnn_5_6_angular_error), 0.0,55.0, linestyles ="dashed", colors ="cyan" )
    #plt.vlines( -1.0, 0.0,55.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,55.0 )
    plt.legend( )
    
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_models_5_6_GeV', dpi=600)
    plt.close()
    
    ax =  format_plot(1)
    plt.hist(table_6_7_angular_error, bins, alpha=0.8, color='black', edgecolor = 'black', histtype='step', label='Table-based')
    for j in range(len(models)):
        cnn_6_7_angular_error = []
        if type == 'track':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_millipede_info_cnn( models[j])

        if type == 'cascade':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_nue_info_cnn(models[j], True )

        for i in range( len(cnn_mc_energy )):
            if (1.0e6 < cnn_mc_energy[i]) and (cnn_mc_energy[i] <= 1.0e7):
                cnn_6_7_angular_error.append( cnn_angular_error[i] )
        
        plt.hist(cnn_6_7_angular_error, bins, alpha=0.35, histtype='step', label='CNN-based')
        
    plt.xlim(0.0, 60.0)
    plt.xlabel('Angular Error [deg.]')
    plt.title(title_str + ' (1.0e6 - 1.0e7 GeV)' )
    #plt.vlines(np.median( table_6_7_angular_error), 0.0,60.0, linestyles ="dashed", colors ="orange")
    #plt.vlines(np.median( cnn_6_7_angular_error), 0.0,60.0, linestyles ="dashed", colors ="cyan" )
    #plt.vlines( -1.0, 0.0,60.0, linestyles ="dashed", colors ="black", label='Median' )
    plt.ylim(0.0,60.0 )
    plt.legend( )

    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str + '_models_6_7_GeV', dpi=600)
    plt.close()




# nue_table_mc_charge, nue_cnn_mc_charge, nue_table_mct_angular_error, nue_cnn_mct_angular_error
def plot_angular_error_charges_curve( models, type, title_str, save_str, energy_range_low,energy_range_high):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
        
    if type == 'track':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_millipede_info_table()

    if type == 'cascade':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_nue_info_table(True )
    
    for i in range(len(bins) - 1):
        table_angular_dist = []
        for j in range(len( table_mc_charge )):
            if (bins[i] < table_mc_charge[j]) and (table_mc_charge[j] <= bins[i+1]):
                table_angular_dist.append( table_angular_error[j] )
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_angular_dist )
        
        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)

    curve_points = np.logspace(3,7, 1000)
    curve_vals = []
    for i in range(len(curve_points)):
        curve_vals.append(np.sqrt( curve_points[i]) )

    ax =  format_plot(1)
    plt.xlabel('Deposited Charge [p.e.]')
    plt.ylabel('Angular Error [deg.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='black', alpha=0.1, label='Table-based Quartiles')
    #plt.plot(bins_mids,table_50_quartiles, color='black', label='Table-based Quartiles')
    
    
    for k in range(len( models)):
        cnn_25_quartiles = []
        cnn_50_quartiles = []
        cnn_75_quartiles = []

        if type == 'track':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_millipede_info_cnn( models[k])

        if type == 'cascade':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_nue_info_cnn(models[k], True )

        for i in range(len(bins) - 1):
            cnn_angular_dist = []

            for j in range(len( cnn_mc_charge )):
                if (bins[i] < cnn_mc_charge[j]) and (cnn_mc_charge[j] <= bins[i+1]):
                    cnn_angular_dist.append( cnn_angular_error[j] )

            cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_angular_dist )

            cnn_25_quartiles.append( cnn_25_quartile)
            cnn_50_quartiles.append( cnn_50_quartile)
            cnn_75_quartiles.append( cnn_75_quartile)

        plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, alpha=0.2, label='CNN-based Quartiles, Model ' + str(k))
        #plt.plot(bins_mids,cnn_50_quartiles, color='cyan', label='CNN-based Quartiles')
        
    plt.legend(loc='upper right' )
    plt.xscale('log')
    #plt.ylim(0.0, 180.0)
    plt.title(title_str )
    plt.xlim(bins_mids[0], bins_mids[-1] )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()





def plot_runtime_energy_curve( models, type ,title_str, save_str, energy_range_low, energy_range_high ):

    bins = np.logspace(np.log10(energy_range_low),np.log10(energy_range_high), 10)
    bins_mids = []
    for i in range( len(bins) - 1 ):
        bins_mids.append( (bins[i] + bins[i + 1])/2.0  )

    table_25_quartiles = []
    table_50_quartiles = []
    table_75_quartiles = []
        
    if type == 'track':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_millipede_info_table()

    if type == 'cascade':
        table_mc_energy, table_mc_charge, table_reco_energy, table_angular_error, table_angular_error_mct_seed, table_time = get_nue_info_table(True )
    
    for i in range(len(bins) - 1):
        table_time_dist = []
        for j in range(len( table_mc_energy )):
            if (bins[i] < table_mc_energy[j]) and (table_mc_energy[j] <= bins[i+1]):
                table_time_dist.append( table_time[j] )
    
        table_25_quartile, table_50_quartile, table_75_quartile = get_quartiles( table_time_dist )

        table_25_quartiles.append( table_25_quartile)
        table_50_quartiles.append( table_50_quartile)
        table_75_quartiles.append( table_75_quartile)

    ax =  format_plot(1)
    plt.xlabel('Primary Energy (MC Truth) [GeV]')
    plt.ylabel('Runtime [sec.]')
    plt.fill_between(bins_mids,table_75_quartiles,table_25_quartiles, color='black', alpha=0.1, label='Table-based Quartiles')
        
    for k in range(len(models)):
        cnn_25_quartiles = []
        cnn_50_quartiles = []
        cnn_75_quartiles = []
        
        if type == 'track':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_millipede_info_cnn( models[j])

        if type == 'cascade':
            cnn_mc_energy, cnn_mc_charge, cnn_reco_energy, cnn_angular_error, cnn_angular_error_mct_seed, cnn_time = get_nue_info_cnn(models[j], True )

        for i in range(len(bins) - 1):
            cnn_time_dist = []
            for j in range(len( cnn_mc_energy )):
                if (bins[i] < cnn_mc_energy[j]) and (cnn_mc_energy[j] <= bins[i+1]):
                    cnn_time_dist.append( cnn_time[j] )
                    
            cnn_25_quartile, cnn_50_quartile, cnn_75_quartile = get_quartiles( cnn_time_dist )
            
            cnn_25_quartiles.append( cnn_25_quartile)
            cnn_50_quartiles.append( cnn_50_quartile)
            cnn_75_quartiles.append( cnn_75_quartile)
        
        plt.fill_between(bins_mids,cnn_75_quartiles,cnn_25_quartiles, alpha=0.2,label='CNN-based Quartiles, Model' + str(k) )
    
    plt.legend(loc='upper right' )
    plt.xscale('log')
    plt.title(title_str )
    plt.savefig('/mnt/home/kochocki/egen_lite/plots/' + save_str , dpi=600)
    plt.close()



plot_angular_error_dist( modelnames, 'cascade', 'Monopod', 'monopod_binned_angular_dists')
plot_angular_error_charges_curve( modelnames, 'cascade', 'Monopod', 'monopod_angular_dist_w_charge', 1.0e2,1.0e7)
plot_runtime_energy_curve( modelnames, 'cascade' ,'Monopod', 'monopod_runtime_dists', 1.0e2, 1.0e7 )

#plot_angular_error_dist( modelnames, 'track', 'Millipede', 'millipede_binned_angular_dists')
#plot_angular_error_charges_curve( modelnames, 'track', 'Millipede', 'millipede_angular_dist_w_charge', 1.0e2,1.0e7)
#plot_runtime_energy_curve( modelnames, 'track' ,'Millipede', 'millipede_runtime_dists', 1.0e2, 1.0e7 )
