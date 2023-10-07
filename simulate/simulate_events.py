#!/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python3

import os,sys
from os.path import expandvars
import logging
import math
import numpy as np
from I3Tray import *
from icecube import (icetray, dataio, dataclasses,interfaces)
import argparse
from optparse import OptionParser
from configparser import ConfigParser
from icecube import (icetray, dataio, dataclasses,interfaces, trigger_sim)
from icecube.dataclasses import I3OMGeo
from icecube import (neutrino_generator,LeptonInjector, earthmodel_service, PROPOSAL, cmc)
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
from icecube import DOMLauncher
from icecube import icetray, dataio, dataclasses, phys_services
from icecube import gulliver, lilliput, gulliver_modules, linefit
from icecube import interfaces, simclasses, sim_services, clsim
from icecube.clsim.traysegments import I3CLSimMakePhotons, I3CLSimMakeHitsFromPhotons
from icecube import trigger_splitter
from icecube import vuvuzela
from icecube.MuonGun import load_model, Floodlight, StaticSurfaceInjector, Cylinder, OffsetPowerLaw, ExtrudedPolygon
from icecube.MuonGun.segments import GenerateBundles
from icecube.sim_services import I3ParticleTypePropagatorServiceMap
from icecube.PROPOSAL import I3PropagatorServicePROPOSAL
from icecube.cmc import I3CascadeMCService
from icecube import PROPOSAL
from icecube import WaveCalibrator
from icecube import wavedeform
from icecube.BadDomList import BadDomListTraySegment
from icecube import DomTools


gcddir = '/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd'

parser = argparse.ArgumentParser()
parser.add_argument("-g","--gcdfile",type=str,help="GCD-File",required=True,dest="gcdfile")
parser.add_argument("-o","--outputfile",type=str,help="Output file",default="",dest="outputfile")
parser.add_argument("--mjd",default="62502",type=str,help='MJD for the GCD file',dest="mjd")
parser.add_argument("--RunId",default=0,type=int,help="Configure run ID",dest="runid")
parser.add_argument("--RNGSeed",default=0,type=int,help="RNG Seed",dest="rngseed")
parser.add_argument("--RNGStream",default=0,type=int,help="RNGStream",dest="rngstream")
parser.add_argument("--RNGNumberOfStreams",default=1,type=int,help="RNGNumberOfStreams",dest="rngnumberofstreams")
parser.add_argument("--MuonGun",  default=False, type=bool, dest="onlyMuonGun")
parser.add_argument("--ParamsMap",default=dict(),help='any other parameters',dest="paramsmap")

args = parser.parse_args()

name = ""
outputfile = args.outputfile + str(os.getenv('SLURM_ARRAY_TASK_ID'))
runid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
rngseed = int(os.getenv('SLURM_ARRAY_TASK_ID'))
file_num = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
file_num_str = str(file_num)


try:
    import json
except ImportError:
    json = None
if isinstance(args.paramsmap,str):
    if not json:
        raise Exception('ParamsMap provided as string, but python does not understand json')
    args.paramsmap = json.loads(args.paramsmap)


event_id = 1
def get_header(frame):
    global event_id
    header          = dataclasses.I3EventHeader()
    header.event_id = event_id
    header.run_id   = rngseed
    frame["I3EventHeader"] = header
    event_id += 1

t = I3Tray()
randomService = I3SPRNGRandomService(rngseed, args.rngnumberofstreams, args.rngstream)
t.context['I3RandomService'] = randomService
t.context['I3FileStager'] = dataio.get_stagers()

def make_propagators(seed):
    propagators = I3ParticleTypePropagatorServiceMap()
    muprop = I3PropagatorServicePROPOSAL()
    cprop = I3CascadeMCService(I3GSLRandomService(seed)) # dummy RNG
    for pt in 'MuMinus', 'MuPlus':
        propagators[getattr(dataclasses.I3Particle.ParticleType, pt)] = muprop
    for pt in 'DeltaE', 'Brems', 'PairProd', 'NuclInt', 'Hadrons', 'EMinus', 'EPlus':
        propagators[getattr(dataclasses.I3Particle.ParticleType, pt)] = cprop
    return propagators


if(args.onlyMuonGun):
    print('Running MuonGun')
    type_str = 'MuonGun'

    CylinderLength = 1200.0*I3Units.m
    CylinderRadius = 700.0*I3Units.m
    CylinderCenter = dataclasses.I3Position(0.0*I3Units.m, 0.0*I3Units.m, 0.0*I3Units.m)
    PowerLawGamma = 1.0
    PowerLawOffset = 1000.0*I3Units.GeV
    PowerLawEmin = 1.0e4*I3Units.GeV
    PowerLawEmax = 1.0e8*I3Units.GeV
 
    surface = Cylinder(CylinderLength, CylinderRadius, CylinderCenter)
    spectrum = OffsetPowerLaw(PowerLawGamma, PowerLawOffset, PowerLawEmin, PowerLawEmax)
    generator = Floodlight(surface, spectrum, math.cos(180.0), math.cos(0.0)) # math.cos(args.zenithmax), math.cos(args.zenithmin))
    
    t.Add(GenerateBundles,"muons", Generator=generator,
             RunNumber=runid, NEvents=5,
             GCDFile=args.gcdfile,
             FromTime=dataclasses.I3Time(55380),
             ToTime=dataclasses.I3Time(55380))
    
    t.AddModule('I3PropagatorModule', 'propagator',PropagatorServices=make_propagators(rngseed),RandomService=randomService, RNGStateName="RNGState")
    


else:
    print('Running LeptonInjector')
    type_str = 'NuE'
    
    t.AddService("I3EarthModelServiceFactory", "Earth")
    t.AddModule("I3InfiniteSource", "source",
        prefix = args.gcdfile,
        stream = icetray.I3Frame.DAQ)

    time = dataclasses.I3Time()
    time.set_mod_julian_time(int(args.mjd), 0, 0)
    
    xs_folder = "/cvmfs/icecube.opensciencegrid.org/data/neutrino-generator/cross_section_data/csms_differential_v1.0/"

    t.AddModule("VolumeLeptonInjector",
        EarthModel      = "Earth",
        Nevents         = 25,
        FinalType1      = dataclasses.I3Particle.ParticleType.EMinus,
        FinalType2      = dataclasses.I3Particle.ParticleType.Hadrons,
        DoublyDifferentialCrossSectionFile  = xs_folder + "/dsdxdy_nu_CC_iso.fits",
        TotalCrossSectionFile               = xs_folder + "/sigma_nu_CC_iso.fits",
        MinimumEnergy   = 1.0e4 * I3Units.GeV,
        MaximumEnergy   = 1.0e7 * I3Units.GeV,
        MinimumZenith   = 0.0,
        MaximumZenith   = 180.0*np.pi/180.0,
        PowerLawIndex   = 1.0,
        CylinderRadius  = 700.0 * I3Units.meter,
        CylinderHeight  = 1200.0 * I3Units.meter,
        MinimumAzimuth  = 0.0 ,
        MaximumAzimuth  = 2.0*np.pi ,
        RandomService   = "I3RandomService")

    t.AddModule(get_header, streams = [icetray.I3Frame.DAQ])
   
    t.AddModule('I3PropagatorModule', 'propagator',PropagatorServices=make_propagators(rngseed),RandomService=randomService, RNGStateName="RNGState")

photon_kwargs = dict(
        GCDFile = args.gcdfile,
        PhotonSeriesName = "I3PhotonSeriesMap",
        MCPESeriesName = None, # turn off conversion to MCPESeries at this step
        MCTreeName = "I3MCTree",
        OutputMCTreeName = None,
        RandomService = randomService,
        UseGPUs=True,
        UseCPUs=False,
        UnWeightedPhotons=False,
        UnWeightedPhotonsScalingFactor=None,
        DOMOversizeFactor=1.0,
        WavelengthAcceptance=None,
        IceModelLocation=expandvars(os.path.join("$I3_SRC/ice-models/resources/models/ICEMODEL/", "spice_ftp-v2")), # Fix ice model!!
        DisableTilt=False,
        UseGeant4 = False,
        UseCascadeExtension=False,
        CrossoverEnergyEM = 0.1,
        CrossoverEnergyHadron=30.0,
        StopDetectedPhotons=True,
        HoleIceParameterization=expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/angsens/as.flasher_p1_0.35_p2_0"),
        UnshadowedFraction = 0.99,
        PhotonHistoryEntries=0,
        DoNotParallelize=False,
        UseI3PropagatorService=False) #IgnoreSubdetectors = ['IceTop'],
                       
t.Add(clsim.traysegments.I3CLSimMakePhotons, "make_photons", **photon_kwargs)

t.Add(I3CLSimMakeHitsFromPhotons, "make_hits_from_photons",
                     PhotonSeriesName        = "I3PhotonSeriesMap",
                     MCPESeriesName          = "I3MCPESeriesMap",
                     RandomService           = randomService,
                     DOMOversizeFactor       = 1.0,
                     UnshadowedFraction      = 0.99,
                     HoleIceParameterization = expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/angsens/as.flasher_p1_0.35_p2_0"),
                     GCDFile                 = args.gcdfile,
                     IceModelLocation        = expandvars(
                         os.path.join('$I3_SRC/ice-models/resources/models/ICEMODEL/', "spice_ftp-v2"))) # IgnoreSubdetectors      = ['IceTop', 'HEX']

ThermalRate = 1.73456e-7
DecayRate = 5.6942e-8
ScintillationHits = 8.072
ScintillationMean = 4.395
ScintillationSigma = 1.777

t.AddModule("Inject", "AddNoiseParams",InputNoiseFile = expandvars("$I3_SRC/vuvuzela/resources/data/parameters.dat") )

t.AddModule("Vuvuzela", name+"_vuvuzela",
                       InputHitSeriesMapName  = "",
                       OutputHitSeriesMapName = "I3MCPESeriesMapNoise",
                       StartWindow            = -10*I3Units.microsecond,
                       EndWindow              = 10*I3Units.microsecond,
                       RandomService          = randomService,
                       OMTypes                = [I3OMGeo.IceCube ],
                       SimulateNewDOMs        = True,
                       DisableLowDTCutoff     = True,
                       UseIndividual          = True,
                       DecayRate = DecayRate,
                       ScintillationHits = ScintillationHits,
                       ScintillationMean = ScintillationMean,
                       ScintillationSigma = ScintillationSigma,
                       ThermalRate = ThermalRate ) # DOMsToExclude          = ExcludeList,
                       
t.Add('I3CombineMCPE', "combineMCPE",
                 InputResponses = ["I3MCPESeriesMap", "I3MCPESeriesMapNoise"],
                 OutputResponse = "I3MCPESeriesMapWithNoise")

t.Add("PMTResponseSimulator","PMTResponse_IC86", # Are these approximations okay?
                     Input="I3MCPESeriesMapWithNoise", # "I3MCPESeriesMap"
                     Output="I3MCPulseSeriesMap",
                     MergeHits=True,
                     LowMem = True,
                     EHEApproximation=False,
                     RandomServiceName="I3RandomService")
                     
t.AddModule("DOMLauncher", "DOMLauncher",
                           Input= "I3MCPulseSeriesMap",
                           Output= "InIceRawData", #"InIcePulses",
                           UseTabulatedPT=True,
                           RandomServiceName="I3RandomService")
                           
t.AddSegment(trigger_sim.TriggerSim, "Old_Triggers",
                        gcd_file = dataio.I3File(args.gcdfile),
                        run_id = runid,
                        prune = False,
                        time_shift = True,
                        filter_mode = False)
                           
t.Add(BadDomListTraySegment.BadDomList,Simulation = True, IgnoreNewDOMs = True)

t.AddModule("I3DOMLaunchCleaning", 'baddomclean',
                   InIceOutput='CleanInIceRawData',
                   IceTopOutput='CleanIceTopRawData')
                   
t.AddModule( 'I3DOMLaunchCleaning', 'OfflineLaunchCleaning',
    InIceInput = 'CleanInIceRawData',
    IceTopInput = 'CleanIceTopRawData',
    InIceOutput = 'OfflineCleanInIceRawData',
    IceTopOutput = 'OfflineCleanIceTopRawData',
    FirstLaunchCleaning = False,
    CleanedKeysList = 'BadDomsListSLC')
    
t.AddModule('I3WaveCalibrator', 'wavecal',
    Launches='OfflineCleanInIceRawData',
    Waveforms='CalibratedWaveforms',
    Errata='CalibrationErrata',
    WaveformRange='CalibratedWaveformRange')
    
#t.Add('Delete', keys=['OfflineCleanInIceRawData'])

t.AddModule('I3Wavedeform', name+'wavedeform',
    Waveforms='CalibratedWaveforms',
    WaveformTimeRange='CalibratedWaveformRange',
    Output='IceCubePulses')
    
# DO I NEED THIS
t.AddModule('I3PMTSaturationFlagger', name+'flag_zorchers',
        Waveforms="CalibratedWaveforms",
        Output="SaturationTimes")
        
def convertToVectorOMKey(frame, inputName, outputName):
    if inputName not in frame: return

    keys = []
    origErrata = frame[inputName]
    for key, window in origErrata:
        keys.append(key)
    newObject = dataclasses.I3VectorOMKey(keys)
    frame[outputName] = newObject
        
t.AddModule(convertToVectorOMKey, name+'convertToVectorOMKey',
        Streams=[icetray.I3Frame.DAQ],
        inputName="SaturationTimes",
        outputName="SaturatedDOMs")
        

t.AddModule('I3TriggerSplitter',"trigger_splitter",
        TrigHierName="I3TriggerHierarchy",
        InputResponses=["IceCubePulses"],
        OutputResponses=["SplitInIcePulses"],
        SubEventStreamName="IC86EventStream",
        TriggerConfigIDs=[1006,1007,1011,21001],
        WriteTimeWindow=True)


t.Add('I3Writer', Filename='/mnt/scratch/kochocki/SPICE_FTP_V2_Test_Data/Process_' + type_str + '_' +  file_num_str  + '.i3.zst')

t.AddModule("TrashCan","trashcan")
t.Execute()
t.Finish()


