#!/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python3

import os,sys
from os.path import expandvars
import logging
import math
import numpy as np
import copy
from I3Tray import *
from icecube import (icetray, dataio, dataclasses,interfaces)
from icecube.dataclasses import I3OMGeo
from icecube import (neutrino_generator, earthmodel_service, PROPOSAL, cmc)
from icecube.simprod.segments import GenerateNeutrinos, PropagateMuons
from icecube.MuonGun import load_model, Floodlight, Cylinder, OffsetPowerLaw, ExtrudedPolygon
from icecube.MuonGun.segments import GenerateBundles
from icecube.sim_services import I3ParticleTypePropagatorServiceMap
from icecube.PROPOSAL import I3PropagatorServicePROPOSAL
from icecube.cmc import I3CascadeMCService
from icecube.phys_services import I3SPRNGRandomService, I3GSLRandomService
from icecube.hdfwriter import I3HDFWriter
from icecube import icetray, dataio, dataclasses, phys_services
from icecube import gulliver, lilliput, gulliver_modules, linefit
from icecube import corsika_reader
from icecube import STTools
from icecube import lilliput, linefit, dataclasses, icetray, dataio, phys_services, hdfwriter, gulliver,simclasses, millipede, icetray, photonics_service
try:
    linefit.simple
except AttributeError:
    from icecube import improvedLinefit
    linefit.simple = improvedLinefit.simple
from icecube.common_variables import direct_hits
from icecube.lilliput.segments import I3SinglePandelFitter, I3IterativePandelFitter
from icecube import spline_reco, mue
from icecube.level3_filter_cascade.level3_Recos import CascadeLlhVertexFit
from icecube.STTools.seededRT import I3SeededRTConfiguration
from icecube.STTools.seededRT import I3SeededRTConfigurationService
from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
from icecube.millipede import HighEnergyExclusions
from icecube.icetray import I3Units
import argparse
import time

from icecube.phys_services.which_split import which_split
from icecube.filterscripts import filter_globals
from icecube.dataclasses import I3Double, I3Particle, I3Direction, I3Position, I3VectorI3Particle, I3Constants, I3VectorOMKey, I3RecoPulseSeriesMap
from icecube.icetray import I3Units, I3Frame, I3ConditionalModule, traysegment
from icecube.millipede import MonopodFit, MuMillipedeFit, TaupedeFit, HighEnergyExclusions, MillipedeFitParams
from icecube.level3_filter_cascade.level3_Recos import CascadeLlhVertexFit, SPEFit
from icecube import VHESelfVeto
from icecube import spline_reco, mue
# for level 2
from icecube import STTools, DomTools
from icecube import lilliput, clast, cscd_llh
from icecube.lilliput.segments import I3SinglePandelFitter, I3IterativePandelFitter
# for level 3 muon
from icecube.level3_filter_muon.level3_SplitHiveSplitter import SplitAndRecoHiveSplitter
# for gulliver
from icecube.gulliver_modules import gulliview


file_num = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
file_num_str = str(file_num)
rngseed = int(os.getenv('SLURM_ARRAY_TASK_ID'))
runid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
rngnumberofstreams = 1
rngstream = 0
mjd = 62502

parser = argparse.ArgumentParser()
parser.add_argument("-g","--gcdfile",type=str,help="GCD-File",default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz",dest="gcdfile")
parser.add_argument("-s","--switch",type=str,help="NuE or MuonGun Reco?",required=True,dest="switch")
parser.add_argument("-o","--infile",type=str,help="Infile location?",default="",dest="infile")

args = parser.parse_args()

file = dataio.I3File(str(args.gcdfile))
file.pop_frame()
geometry = file.pop_frame()['I3Geometry']


def get_dep_volume_charge(frame, geometry):

    if 'SplitInIcePulsesLatePulseCleaned' in frame:
        omgeo = geometry.omgeo
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SplitInIcePulsesLatePulseCleaned')
        
        
        #return sep values from IceCube OMs and Gen2 OMs
        
        IceCube_charge = 0.0
        
        for key,pulseList in pulse_map:
            for pulse in pulseList:
                charge = pulse.charge
                string = key.string
                if (string <= 86) and (key.om>=0 and key.om<=60):
                    IceCube_charge = IceCube_charge + charge
                    
        frame['Qtot'] = dataclasses.I3Double(IceCube_charge)

def get_vertex_pos(frame, geometry ):
        
    event_vertex_x = frame['I3MCTree'][0].pos.x
    event_vertex_y = frame['I3MCTree'][0].pos.y
    event_vertex_z = frame['I3MCTree'][0].pos.z
    
    frame['Vertex_X'] = dataclasses.I3Double(event_vertex_x)
    frame['Vertex_Y'] = dataclasses.I3Double(event_vertex_y)
    frame['Vertex_Z'] = dataclasses.I3Double(event_vertex_z)


def get_mc_truth(frame):
    tree = frame['I3MCTree']
    prims = tree.primaries

    initial_neutrino = None
    for p in prims:
        if p.is_neutrino:
            initial_neutrino = p
            p.fit_status = dataclasses.I3Particle.FitStatus.OK
            p.shape = dataclasses.I3Particle.Cascade
            p.type = dataclasses.I3Particle.ParticleType.EMinus
            frame.Put("MC_Truth", p)
            break
            
@traysegment
def Level2ReconstructionWrapper(tray, name, Pulses='SplitInIcePulses'):

    """
    SRT cleaning: standard level 2
    """
    # rename keys from Gen2 BaseProc
    tray.Add('Rename', 'rename',
             Keys=['SRTCleanedInIcePulses', 'SRT'+Pulses,
                    Pulses+'UnmaskedTimeRange', Pulses+'TimeRange'],
             If=lambda frame: frame.Has('SRTCleanedInIcePulses'))
    seededRTConfig = I3DOMLinkSeededRTConfigurationService(
        ic_ic_RTRadius              = 150.0*I3Units.m,
        ic_ic_RTTime                = 1000.0*I3Units.ns,
        useDustlayerCorrection      = False,
        allowSelfCoincidence        = True)
    tray.Add('I3SeededRTCleaning_RecoPulseMask_Module', 'seededrt',
        InputHitSeriesMapName  = Pulses,
        OutputHitSeriesMapName = 'SRT' + Pulses,
        STConfigService        = seededRTConfig,
        SeedProcedure          = 'HLCCoreHits',
        NHitsThreshold         = 2,
        MaxNIterations         = 3,
        Streams                = [I3Frame.Physics],
        If=lambda frame: not frame.Has('SRT'+Pulses))

    """
    offline muon reconstruction of LineFit and SPEFit2: taken from standard level 2 filterscripts
    """
    tray.Add(linefit.simple, inputResponse = 'SRT' + Pulses, fitName = 'LineFit',
             If = lambda frame: not 'LineFit' in frame)
    tray.Add(I3SinglePandelFitter, 'SPEFitSingle',
        fitname = 'SPEFitSingle',
        Pulses = 'SRT' + Pulses,
        seeds = ['LineFit'],
        If = lambda frame : not 'SPEFit2' in frame)
    tray.Add(I3IterativePandelFitter, 'SPEFit2',
        fitname = 'SPEFit2',
        Pulses = 'SRT' + Pulses,
        n_iterations = 2,
        seeds = ['SPEFitSingle'],
        If = lambda frame : not 'SPEFit2' in frame)

    """
    offline cascade reconstruction of CascadeLast and CascadeLlhVertexFit: taken from standard level 2 filterscripts
    """
    tray.Add('I3CLastModule', 'CascadeLast_L2',
        Name = 'CascadeLast_L2',
        InputReadout = Pulses,
        If = lambda frame : not 'CascadeLast_L2' in frame)
    tray.Add('I3CscdLlhModule', 'CascadeLlhVertexFit_L2',
        InputType = 'RecoPulse', # ! Use reco pulses
        RecoSeries = Pulses, # ! Name of input pulse series
        FirstLE = True, # Default
        SeedWithOrigin = False, # Default
        SeedKey = 'CascadeLast_L2', # ! Seed fit - CLast reco
        MinHits = 8, # ! Require 8 hits
        AmpWeightPower = 0.0, # Default
        ResultName = 'CascadeLlhVertexFit_L2', # ! Name of fit result
        Minimizer = 'Powell', # ! Set the minimizer to use
        PDF = 'UPandel', # ! Set the pdf to use
        ParamT = '1.0, 0.0, 0.0, false',   # ! Setup parameters
        ParamX = '1.0, 0.0, 0.0, false',   # ! Setup parameters
        ParamY = '1.0, 0.0, 0.0, false',   # ! Setup parameters
        ParamZ = '1.0, 0.0, 0.0, false',   # ! Setup parameters
        If = lambda frame : not 'CascadeLlhVertexFit_L2' in frame)

    """
    clean up
    """
    deletekeys = ['LineFit_HuberFit', 'LineFit_Pulses_delay_cleaned', 'LineFit_debiasedPulses', 'LineFit_linefit_final_rusage',
                  'SPEFitSingle', 'SPEFitSingleFitParams']
    tray.Add('Delete', Keys=deletekeys)

    
################################################################
############## LEVEL 3 RECONSTRUCTION WRAPPER ##################
################################################################

@traysegment
def Level3ReconstructionWrapper(tray, name, Pulses='SplitInIcePulses'):

    """
    DOM selection: for cascade level 3 reconstructions DeepCore DOMs should be excluded
    """
    tray.Add('I3OMSelection<I3RecoPulseSeries>', 'omselection',
        InputResponse = 'SRT' + Pulses,
        OmittedStrings = [79,80,81,82,83,84,85,86],
        OutputOMSelection = 'SRT' + Pulses + '_BadOMSelectionString',
        OutputResponse = 'SRT' + Pulses + '_IC_Singles')

    """
    CascadeLlhVertexFit: standard level 3 fit on SRT Pulses without DeepCore
    """
    tray.Add(CascadeLlhVertexFit, 'CascadeLlhVertexFit_L3',
        Pulses = 'SRT' + Pulses + '_IC_Singles',
        If = lambda frame: not 'CascadeLlhVertexFit_L3' in frame)
    
    """
    SPEFit: standard level 3 fit on SRT pulses without DeepCore (first guesses are SPEFit2 and LineFit from level 2)
    """
    tray.Add(SPEFit, 'SPEFit16',
        Pulses = 'SRT' + Pulses + '_IC_Singles',
        Iterations = 16,
        If = lambda frame: not 'SPEFit16' in frame)

    """
    make the cascade level 3 seed: take the best combination out of all level 2 and level 3 fits to build a seed
    """
    def addlevel3seed(frame, Output):
    
        # the seed particle
        seed = I3Particle()
        seed.pos = I3Position(0, 0, 0)
        seed.dir = I3Direction(0, 0)
        seed.time = 0
        seed.energy = 0.
        seed.length = 0.
        seed.speed = I3Constants.c
        seed.fit_status = I3Particle.OK
        seed.shape = I3Particle.Cascade
        seed.location_type = I3Particle.InIce
        
        # possible solutions (ordered, construct seed in any case, even if level 2 + 3 recos failed)
        vertexfits = ['CascadeLlhVertexFit_L3', 'CascadeLlhVertexFit_L2', 'CascadeLast_L2']
        directionfits = ['SPEFit16', 'SPEFit2', 'LineFit']
        
        # vertex + time
        for vertexfitname in vertexfits:
            if not vertexfitname in frame:
                continue
            vertexfit = frame[vertexfitname]
            if vertexfit.fit_status == I3Particle.OK and vertexfit.pos.r >= 0 and vertexfit.time >= 0:
                seed.pos = vertexfit.pos
                seed.time = vertexfit.time
                break
        
        # direction
        for directionfitname in directionfits:
            if not directionfitname in frame:
                continue
            directionfit = frame[directionfitname]
            if directionfit.fit_status == I3Particle.OK and directionfit.dir.theta >= 0 and directionfit.dir.phi >= 0:
                seed.dir = directionfit.dir
                break
        
        # save it
        frame.Put(Output, seed)

    tray.Add(addlevel3seed, Output=name, If=lambda frame: not name in frame)

    """
    clean up
    """
    deletekeys = ['CascadeLlhVertexFit_L3_CLastSeed', 'CascadeLlhVertexFit_L3_CLastSeedParams']
    deletekeys += ['SRT' + Pulses + '_BadOMSelectionString', 'SRT' + Pulses + '_IC_Singles', 'SRT' + Pulses + '_IC_SinglesCleanedKeys']
    tray.Add('Delete', keys=deletekeys)
    
def _weighted_quantile_arg(values, weights, q=0.5):
    indices = np.argsort(values)
    sorted_indices = np.arange(len(values))[indices]
    medianidx = (weights[indices].cumsum()/weights[indices].sum()).searchsorted(q)
    if (0 <= medianidx) and (medianidx < len(values)):
        return sorted_indices[medianidx]
    else:
        return np.nan

def weighted_quantile(values, weights, q=0.5):
    if len(values) != len(weights):
        raise ValueError("shape of `values` and `weights` don't match!")
    index = _weighted_quantile_arg(values, weights, q=q)
    if not np.isnan(index):
        return values[index]
    else:
        return np.nan

def weighted_median(values, weights):
    return weighted_quantile(values, weights, q=0.5)

def LatePulseCleaning(frame, Pulses, Residual=3e3*I3Units.ns):
    pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, Pulses)
    mask = dataclasses.I3RecoPulseSeriesMapMask(frame, Pulses)
    counter, charge = 0, 0
    qtot = 0
    times = dataclasses.I3TimeWindowSeriesMap()
    for omkey, ps in pulses.items():
        if len(ps) < 2:
            if len(ps) == 1:
                qtot += ps[0].charge
            continue
        ts = np.asarray([p.time for p in ps])
        cs = np.asarray([p.charge for p in ps])
        median = weighted_median(ts, cs)
        qtot += cs.sum()
        ### DEBUG
        # if cs.sum()>200:
        #     from matplotlib import pyplot as plt
        #     plt.figure()
        #     plt.hist(ts, bins=np.arange(median-0.5*Residual, median+3*Residual, 50), weights=cs, histtype='step')
        #     [plt.vlines(_, 0, 10) for _ in [median-Residual, median, median+Residual]]
        #     plt.title(omkey)
        #     plt.yscale('log')
        #     plt.savefig(f'out/misc/pulses/{omkey.string}_{omkey.om}.png')
        for p in ps:
            if p.time >= (median+Residual):
                if not times.has_key(omkey):
                    ts = dataclasses.I3TimeWindowSeries()
                    ts.append(dataclasses.I3TimeWindow(median+Residual, np.inf)) # this defines the **excluded** time window
                    times[omkey] = ts
                mask.set(omkey, p, False)
                counter += 1
                charge += p.charge
    frame[Pulses+"LatePulseCleaned"] = mask
    frame[Pulses+"LatePulseCleanedTimeWindows"] = times
    frame[Pulses+"LatePulseCleanedTimeRange"] = copy.deepcopy(frame[Pulses+"TimeRange"])




def start_time( frame):
    frame['Start_Reco_Time'] = dataclasses.I3Double( time.time() )
    
def end_time( frame):
    frame['End_Reco_Time'] = dataclasses.I3Double( time.time() )


PulsesForReco = 'SRTCleanedInIcePulses'
UncleanedPulses = 'SplitInIcePulses'
FitPrefix = ""

muon_service = None
effective_distance = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_effectivedistance_spice_ftp-v1_z20.eff.fits'
table_base = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_single_spice_ftp-v1_flat_z20_a5.%s.fits')
tilttabledir = os.path.expandvars('/mnt/home/kochocki/egen_lite/icetray/photonics-service/resources/tilt/')
cascade_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', timingSigma=0.0, effectivedistancetable=effective_distance, tiltTableDir=tilttabledir)

tray = I3Tray()
randomService = phys_services.I3SPRNGRandomService(rngseed, rngnumberofstreams, rngstream)
tray.context['I3RandomService'] = randomService
tray.context['I3FileStager'] = dataio.get_stagers()

tray.Add("I3Reader", FilenameList=[args.gcdfile, '/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Process_' + args.switch + '_' +  file_num_str  + '.i3.zst' ] )

tray.AddModule('I3LCPulseCleaning', 'cleaning',
       OutputHLC=UncleanedPulses+'HLC',
       OutputSLC='', Input=UncleanedPulses,
       If=lambda frame: not frame.Has(UncleanedPulses+'HLC'))
       
tray.Add(Level2ReconstructionWrapper,'level2reco',Pulses = UncleanedPulses)

tray.Add(Level3ReconstructionWrapper,'CombinedCascadeSeed_L3',Pulses = UncleanedPulses)
             
'''
stConfigService = I3DOMLinkSeededRTConfigurationService(
    allowSelfCoincidence    = True,           # Default
    useDustlayerCorrection  = True,            # Default
    dustlayerUpperZBoundary = 0*I3Units.m,     # Default
    dustlayerLowerZBoundary = -150*I3Units.m,  # Default
    ic_ic_RTTime            = 1000*I3Units.ns, # Default
    ic_ic_RTRadius          = 150*I3Units.m    # Default )

tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'seededRTcleaning',
        InputHitSeriesMapName  = "SplitInIcePulses",
        OutputHitSeriesMapName = "SRTCleanedInIcePulses",
        STConfigService        =  stConfigService,
        SeedProcedure          = 'HLCCoreHits',
        NHitsThreshold         = 2,
        MaxNIterations         = 3,
        Streams                = [icetray.I3Frame.Physics] )
'''

tray.AddModule(get_mc_truth, 'get_mc_truth', Streams=[icetray.I3Frame.Physics] )

exclusions = tray.AddSegment(HighEnergyExclusions, Pulses=UncleanedPulses, BadDomsList='BadDomsList', ExcludeBrightDOMs=False, CalibrationErrata = 'CalibrationErrata',  BrightDOMThreshold = 15, ExcludeSaturatedDOMs = 'SaturatedDOMs', SaturationWindows = 'SaturationWindows')

millipede_params = {'ExcludedDOMs': exclusions, 'ReadoutWindow': 'SplitInIcePulsesTimeRange', 'PartialExclusion': True, 'UseUnhitDOMs': True, 'RelUncertainty': 0.05}
    
tray.AddModule(LatePulseCleaning, "LatePulseCleaning",
                   Pulses=UncleanedPulses, Residual=1500*I3Units.ns,
                   If=lambda frame: not frame.Has(UncleanedPulses+'LatePulseCleaned'))
                   
exclusions.append(UncleanedPulses+'LatePulseCleanedTimeWindows')
                   
millipede_params['Pulses'] = UncleanedPulses+'LatePulseCleaned'

millipede_params['ReadoutWindow'] = UncleanedPulses+'LatePulseCleanedTimeRange'

'''
How much does seed to MonopodFit first fit matter, CascadeLlhVertexFit okay?
Do Monopod pulses need late pulse cleaning?
Difference between iMIGRAD and MIGRAD?
Tianlu's settings have PhotonsPerBin=0?
SRT Cleaning Settings? Allowing self-coincidence
'''

tray.AddModule(lambda frame: 'CombinedCascadeSeed_L3' in frame, 'seed_exists')

tray.AddModule(start_time, 'start_time', Streams=[icetray.I3Frame.Physics]  )
    
tray.AddSegment(millipede.MonopodFit, 'Monopod_Amplitude',
    CascadePhotonicsService=cascade_service,
    Seed='CombinedCascadeSeed_L3',
    Iterations=1,
    Minimizer='MIGRAD',
    PhotonsPerBin=-1,
    **millipede_params)
    
tray.AddModule(lambda frame: 'CombinedCascadeSeed_L3' in frame, 'amplitude_seed_exists')
    
tray.AddSegment(millipede.MonopodFit, 'Monopod_5photons_4iterations',
    CascadePhotonicsService=cascade_service,
    Seed='Monopod_Amplitude',
    Iterations=4,
    Minimizer='MIGRAD',
    PhotonsPerBin=5,
    **millipede_params)
    
tray.AddModule(end_time, 'end_time', Streams=[icetray.I3Frame.Physics]  )
    
tray.AddSegment(millipede.MonopodFit, 'Monopod_0photons_4iterations',
    CascadePhotonicsService=cascade_service,
    Seed='Monopod_Amplitude',
    Iterations=4,
    Minimizer='MIGRAD',
    PhotonsPerBin=0,
    **millipede_params)

tray.AddSegment(millipede.MonopodFit, 'Monopod_5photons_4iterations_MCTSeed',
    CascadePhotonicsService=cascade_service,
    Seed='MC_Truth', #Seed='MC_Truth',
    Iterations=4,
    Minimizer='MIGRAD',
    PhotonsPerBin=5,
    If=lambda frame: 'MC_Truth' in frame,
    **millipede_params)
    
tray.AddSegment(millipede.MonopodFit, 'Monopod_0photons_4iterations_MCTSeed',
    CascadePhotonicsService=cascade_service,
    Seed='MC_Truth', #Seed='MC_Truth',
    Iterations=4,
    Minimizer='MIGRAD',
    PhotonsPerBin=0,
    If=lambda frame: 'MC_Truth' in frame,
    **millipede_params)


tray.AddModule(get_dep_volume_charge, 'get_dep_volume_charge', geometry=geometry, Streams=[icetray.I3Frame.Physics] )

tray.AddModule(get_vertex_pos, 'get_vertex_pos', geometry=geometry, Streams=[icetray.I3Frame.Physics] )

if os.path.isfile('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst'):
    os.remove('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst')

if os.path.isfile('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5'):
    os.remove('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5')

tray.AddModule('I3Writer', 'writer', filename='/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst')

tray.AddSegment(I3HDFWriter,output='/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5',
    keys= ['Monopod_5photons_4iterations',
    'Monopod_5photons_4iterationsFitParams',
    'Monopod_5photons_4iterations_MCTSeed',
    'Monopod_5photons_4iterations_MCTSeedFitParams',
    'Monopod_0photons_4iterations_MCTSeed',
    'Monopod_0photons_4iterations_MCTSeedFitParams',
    'Monopod_0photons_4iterations',
    'Monopod_0photons_4iterationsFitParams',
    'Monopod_Amplitude',
    'Monopod_AmplitudeFitParams',
    'I3EventHeader',
    'I3MCTree',
    'MC_Truth',
    'Qtot',
    'Vertex_X',
    'Vertex_Y',
    'Vertex_Z',
    'Start_Reco_Time',
    'End_Reco_Time'], SubEventStreams=['IC86EventStream'] )

tray.AddModule("TrashCan","trashcan")
tray.Execute()
tray.Finish()

