#!/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python3

import os,sys
from os.path import expandvars
import logging
import math
import numpy as np
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

    if 'SRTCleanedInIcePulses' in frame:
        omgeo = geometry.omgeo
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SRTCleanedInIcePulses')
        
        
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
            p.shape = dataclasses.I3Particle.InfiniteTrack
            p.type = dataclasses.I3Particle.ParticleType.MuMinus
            frame.Put("MC_Truth", p)
            break
            
def start_time( frame):
    frame['Start_Reco_Time'] = dataclasses.I3Double( time.time() )
    
def end_time( frame):
    frame['End_Reco_Time'] = dataclasses.I3Double( time.time() )

PulsesForReco = 'SRTCleanedInIcePulses'
FitPrefix = ""

muon_service = None
effective_distance = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_effectivedistance_spice_ftp-v1_z20.eff.fits'
table_base = os.path.expandvars('/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/cascade_single_spice_ftp-v1_flat_z20_a5.%s.fits')
cascade_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', timingSigma=0.0, effectivedistancetable=effective_distance)


if os.path.exists('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst'):
    os.remove('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst')

if os.path.exists('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5'):
    os.remove('/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5')


tray = I3Tray()
randomService = phys_services.I3SPRNGRandomService(rngseed, rngnumberofstreams, rngstream)
tray.context['I3RandomService'] = randomService
tray.context['I3FileStager'] = dataio.get_stagers()

tray.Add("I3Reader", FilenameList=[args.gcdfile, '/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Process_' + args.switch + '_' +  file_num_str  + '.i3.zst' ] )

tray.AddModule(get_mc_truth, 'get_mc_truth', Streams=[icetray.I3Frame.Physics] )

stConfigService = I3DOMLinkSeededRTConfigurationService(
    allowSelfCoincidence    = False,           # Default
    useDustlayerCorrection  = True,            # Default
    dustlayerUpperZBoundary = 0*I3Units.m,     # Default
    dustlayerLowerZBoundary = -150*I3Units.m,  # Default
    ic_ic_RTTime            = 1000*I3Units.ns, # Default
    ic_ic_RTRadius          = 150*I3Units.m    # Default
)


tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'seededRTcleaning',
        InputHitSeriesMapName  = "SplitInIcePulses",
        OutputHitSeriesMapName = "SRTCleanedInIcePulses",
        STConfigService        =  stConfigService,
        SeedProcedure          = 'HLCCoreHits',
        NHitsThreshold         = 2,
        MaxNIterations         = 3,
        Streams                = [icetray.I3Frame.Physics] )

from icecube.icetop_Level3_scripts.modules import AddReadoutTimeWindow
#tray.AddModule(lambda frame: frame.Has('SplitInIcePulses') , 'Cleaned_event')
#tray.AddModule(AddReadoutTimeWindow,'Add_Window', Pulses='SplitInIcePulses' )


tray.AddService("I3GulliverMinuitFactory", "default_simplex",
        Algorithm="SIMPLEX",
        MaxIterations=1000,
        Tolerance=0.01)
tray.AddService("I3SimpleParametrizationFactory", "default_simpletrack",
        StepX = 20*I3Units.m,
        StepY = 20*I3Units.m,
        StepZ = 20*I3Units.m,
        StepZenith = 0.1*I3Units.radian,
        StepAzimuth= 0.2*I3Units.radian,
        BoundsX = [-5000*I3Units.m, 5000*I3Units.m],
        BoundsY = [-5000*I3Units.m, 5000*I3Units.m],
        BoundsZ = [-5000*I3Units.m, 5000*I3Units.m])
tray.AddSegment(linefit.simple, 'LineFit',
                inputResponse = PulsesForReco,
                fitName = FitPrefix+'LineFit')
tray.AddSegment(I3SinglePandelFitter, FitPrefix+'SPEFitSingle',
                pulses = PulsesForReco,
                seeds = ['LineFit'],
                fitname = 'SPEFitSingle')
tray.AddSegment(I3IterativePandelFitter,
                FitPrefix+'SPEFit8',
                pulses = PulsesForReco,
                n_iterations = 8,
                seeds = ['SPEFitSingle'],
                fitname = 'SPEFit8')
tray.AddModule("muex", "muex_angular4",
               pulses=PulsesForReco,
               rectrk="",
               result=FitPrefix+'MuEXAngular4',
               lcspan=0,
               repeat=4,
               usempe=True,
               detail=False,
               energy=False,
               icedir=os.path.expandvars("$I3_BUILD/mue/resources/ice/mie"))
tray.Add(I3IterativePandelFitter, FitPrefix+"MPEFit4",
            pulses=PulsesForReco,
            domllh="MPE",
            seeds=["MuEXAngular4"],
            n_iterations=4,
            fitname = "MPEFit4")
splinempe_kwargs = dict(
    PulsesName=PulsesForReco,
    TrackSeedList=["MuEXAngular4"],
    BareMuTimingSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_prob_z20a10_V2.fits',
    BareMuAmplitudeSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_abs_z20a10_V2.fits',
    StochTimingSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfHighEStoch_mie_prob_z20a10.fits',
    StochAmplitudeSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfHighEStoch_mie_abs_z20a10.fits')
tray.Add(spline_reco.SplineMPE,FitPrefix+"SplineMPE_recommended",
            configuration="recommended",
            EnergyEstimators=[FitPrefix+"SplineMPEMuEXDifferential"], fitname='SplineMPE_recommended',
            **splinempe_kwargs)

tray.AddService('I3GSLSimplexFactory', 'minimizeit', Tolerance=1e-3, SimplexTolerance=1e-3, MaxIterations=50000)
tray.AddService('MuMillipedeParametrizationFactory', 'millipede_param',
    StepX=5*I3Units.m,
    StepY=5*I3Units.m,
    StepZ=5*I3Units.m,
    StepT=5*I3Units.ns,
    StepZenith=.1*I3Units.deg,
    StepAzimuth=.1*I3Units.deg,
    StepLogE=0,
    MuonSpacing=0,
    ShowerSpacing=10)
tray.AddService('I3BasicSeedServiceFactory', 'seed_millipede', FirstGuess='SplineMPE_recommended')
kwargs = dict(ReadoutWindow='SplitInIcePulsesTimeRange')
millipede_config = dict(MuonPhotonicsService=None,
                    CascadePhotonicsService=cascade_service,
                    Pulses='SplitInIcePulses',
                    ExcludedDOMs=list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
                    PhotonsPerBin=10)  #MuonEnergyLosses=0, #LogDerivative=0  ) # could change excluded doms to copy andrew
millipede_config.update(kwargs)
tray.AddService('MillipedeLikelihoodFactory', 'millipede_likelihood', **millipede_config)
tray.AddModule('I3SimpleFitter', 'LLHFit_Millipede',
    SeedService='seed_millipede',
    Parametrization='millipede_param',
    LogLikelihood='millipede_likelihood',
    Minimizer='minimizeit',
    If=lambda frame: 'SplitInIcePulses' in frame,
    OutputName="FirstMillipedeFit_10mSpacing")
    
tray.AddService('MuMillipedeParametrizationFactory', 'millipede_param2',
    StepX=5*I3Units.m,
    StepY=5*I3Units.m,
    StepZ=5*I3Units.m,
    StepT=5*I3Units.ns,
    StepZenith=.1*I3Units.deg,
    StepAzimuth=.1*I3Units.deg,
    StepLogE=1,
    MuonSpacing=0,
    ShowerSpacing=1,
    BoundsLogE=[3,9])
tray.AddService('I3BasicSeedServiceFactory', 'seed_millipede2', FirstGuess='FirstMillipedeFit_10mSpacing')
tray.AddService('MillipedeLikelihoodFactory', 'millipede_likelihood2',
    CascadePhotonicsService=cascade_service,
    MuonPhotonicsService=None,
    PhotonsPerBin=10,
    ExcludedDOMs = list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
    Pulses='SplitInIcePulses',
    ReadoutWindow='SplitInIcePulsesTimeRange')
    
tray.AddModule(start_time, 'start_time', Streams=[icetray.I3Frame.Physics]  )

tray.AddModule('I3SimpleFitter', 'LLHFit_Millipede2',
    SeedService='seed_millipede2',
    Parametrization='millipede_param2',
    LogLikelihood='millipede_likelihood2',
    Minimizer='minimizeit',
    If=lambda frame: 'SplitInIcePulses' in frame,
    OutputName="SecondMillipedeFit_1mSpacing")
    
tray.AddModule(end_time, 'end_time', Streams=[icetray.I3Frame.Physics]  )

# this performs the actual fit, I3SimpleFitter

# Everything below, MuMillipede, assumes the same direction, and re-solves the losses to minimize likelihood. Fits below give likelihoods at SeedTrack direction

# Want to look at output of first

tray.AddModule('MuMillipede', 'millipede_fit',
    CascadePhotonicsService=cascade_service,
    MuonPhotonicsService=None,
    PhotonsPerBin=10,
    MuonRegularization=0,
    ShowerRegularization=0,
    ShowerSpacing=10,
    MuonSpacing=0,
    ExcludedDOMs = list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
    Pulses='SplitInIcePulses',
    ReadoutWindow='SplitInIcePulsesTimeRange',
    SeedTrack='FirstMillipedeFit_10mSpacing',
    If = lambda fr: 'SecondMillipedeFit_1mSpacing' in fr,
    Output='FirstMillipedeFit_10mSpacing_MillipedeSeed')

tray.AddModule('MuMillipede', 'millipede_fit2',
    CascadePhotonicsService=cascade_service,
    MuonPhotonicsService=None,
    PhotonsPerBin=10,
    MuonRegularization=0,
    ShowerRegularization=0,
    ShowerSpacing=1,
    MuonSpacing=0,
    ExcludedDOMs = list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
    Pulses='SplitInIcePulses',
    ReadoutWindow='SplitInIcePulsesTimeRange',
    SeedTrack='SecondMillipedeFit_1mSpacing',
    If = lambda fr: 'SecondMillipedeFit_1mSpacing' in fr,
    Output='SecondMillipedeFit_1mSpacing_MillipedeSeed'
    )
    
tray.AddModule('MuMillipede', 'millipede_fit_MCTSeed',
    CascadePhotonicsService=cascade_service,
    MuonPhotonicsService=None,
    PhotonsPerBin=10,
    MuonRegularization=0,
    ShowerRegularization=0,
    ShowerSpacing=10,
    MuonSpacing=0,
    ExcludedDOMs = list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
    Pulses='SplitInIcePulses',
    ReadoutWindow='SplitInIcePulsesTimeRange',
    SeedTrack='MC_Truth',
    If = lambda fr: ('SecondMillipedeFit_1mSpacing' in fr) and ('MC_Truth' in fr),
    Output='FirstMillipedeFit_10mSpacing_MCTSeed')

tray.AddModule('MuMillipede', 'millipede_fit2_MCTSeed',
    CascadePhotonicsService=cascade_service,
    MuonPhotonicsService=None,
    PhotonsPerBin=10,
    MuonRegularization=0,
    ShowerRegularization=0,
    ShowerSpacing=1,
    MuonSpacing=0,
    ExcludedDOMs = list(set(['CalibrationErrata', 'SaturationWindows'] + ["BadDomsList"])),
    Pulses='SplitInIcePulses',
    ReadoutWindow='SplitInIcePulsesTimeRange',
    SeedTrack='MC_Truth',
    If = lambda fr: ('SecondMillipedeFit_1mSpacing' in fr) and ('MC_Truth' in fr),
    Output='SecondMillipedeFit_1mSpacing_MCTSeed')



tray.AddModule(get_dep_volume_charge, 'get_dep_volume_charge', geometry=geometry, Streams=[icetray.I3Frame.Physics] )

tray.AddModule(get_vertex_pos, 'get_vertex_pos', geometry=geometry, Streams=[icetray.I3Frame.Physics] )


tray.AddModule('I3Writer', 'writer', filename='/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.i3.zst')

tray.AddSegment(I3HDFWriter,output='/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Table_Reco_' + args.switch + '_' +  file_num_str  + '.hdf5',
    keys= ['FirstMillipedeFit_10mSpacing_MillipedeSeed',
    'FirstMillipedeFit_10mSpacing_MillipedeSeedFitParams',
    'FirstMillipedeFit_10mSpacing_MCTSeed',
    'FirstMillipedeFit_10mSpacing_MCTSeedFitParams',
    'FirstMillipedeFit_10mSpacing',
    'FirstMillipedeFit_10mSpacingFitParams',
    'FirstMillipedeFit_10mSpacingParams',
    'SecondMillipedeFit_1mSpacing_MillipedeSeed',
    'SecondMillipedeFit_1mSpacing_MillipedeSeedFitParams',
    'SecondMillipedeFit_1mSpacing_MCTSeed',
    'SecondMillipedeFit_1mSpacing_MCTSeedFitParams',
    'SecondMillipedeFit_1mSpacing',
    'SecondMillipedeFit_1mSpacingFitParams',
    'SecondMillipedeFit_1mSpacingParams',
    'FirstMillipedeFit_10mSpacing_millipede_likelihood',
    'SecondMillipedeFit_1mSpacing_millipede_likelihood2',
    'I3EventHeader',
    'I3MCTree',
    'MC_Truth',
    'Qtot',
    'LineFit',
    'LineFitParams',
    'SPEFit8',
    'SPEFit8FitParams',
    'Vertex_X',
    'Vertex_Y',
    'Vertex_Z',
    'Start_Reco_Time',
    'End_Reco_Time'], SubEventStreams=['IC86EventStream'] )


tray.AddModule("TrashCan","trashcan")
tray.Execute(5)
tray.Finish()

