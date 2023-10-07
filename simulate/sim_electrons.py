#!/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/bin/python3

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

numevents = 1000 # Previously 10000, reduced to run on scavenger nodes. Combine files later
rngseed = int(os.getenv('SLURM_ARRAY_TASK_ID'))
runid = int(os.getenv('SLURM_ARRAY_TASK_ID'))


np.random.seed(rngseed)
sensors = ['IceCube']
effs = [1.0]
gpu = True
rngnumberofstreams = 1
rngstream = 0
mjd = 62502

def make_propagators(seed):
    propagators = I3ParticleTypePropagatorServiceMap()
    muprop = I3PropagatorServicePROPOSAL()#cylinderHeight=1600.0, cylinderRadius=800.0)
    cprop = I3CascadeMCService(phys_services.I3GSLRandomService(seed)) # dummy RNG
    for pt in 'MuMinus', 'MuPlus':
        propagators[getattr(dataclasses.I3Particle.ParticleType, pt)] = muprop
    for pt in 'DeltaE', 'Brems', 'PairProd', 'NuclInt', 'Hadrons', 'EMinus', 'EPlus':
        propagators[getattr(dataclasses.I3Particle.ParticleType, pt)] = cprop
    return propagators


class Add_MCTree_Lepton(icetray.I3Module):

    count   = 0
    nevents = 0
    
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.AddParameter("NEvents","name of event counter",self.nevents)
        self.AddOutBox("OutBox")
        
    def Configure(self):
        self.nevents = self.GetParameter("NEvents")
        
    def DAQ(self, frame):
        energy =  10.0*I3Units.TeV
        
        cyl_angle = np.random.rand()*np.pi*2.0
        radius = np.sqrt(np.random.rand())*850.0 # IceCube cylinder radius with padding for millipede track parameterization
        z = np.random.rand()*1200.0 - 600.0
        pos_x = radius*np.cos( cyl_angle)
        pos_y = radius*np.sin( cyl_angle)
        pos_z = z
        phi = np.random.rand()*np.pi*2.0
        theta = np.random.rand()*np.pi
        
        lepton = dataclasses.I3Particle()
        lepton.type = dataclasses.I3Particle.ParticleType.EMinus
        lepton.energy = energy
        lepton.pos.x = pos_x*I3Units.m
        lepton.pos.y = pos_y*I3Units.m
        lepton.pos.z = pos_z*I3Units.m
        lepton.time = 0.0*I3Units.ns
        lepton.fit_status = dataclasses.I3Particle.FitStatus.OK
        lepton.location_type = dataclasses.I3Particle.LocationType.InIce
        direction = dataclasses.I3Direction()
        direction.set_theta_phi(theta, phi  )
        lepton.dir = direction
     
        mct = dataclasses.I3MCTree(lepton) #dataclasses.physics.I3MCTree() ?
        frame['I3MCTree'] = mct
        
        self.count += 1
        if self.count < self.nevents:
            self.PushFrame(frame)
        elif self.nevents <= self.count:
            self.PushFrame(frame)
            self.RequestSuspension()

def get_mc_truth(frame):
    tree = frame['I3MCTree']
    lepton = tree[0]
    if not (lepton.type == dataclasses.I3Particle.ParticleType.EMinus):
        print( 'Not EPlus' )
    else:
        frame.Put("LabelsDeepLearning", lepton)
        
def sort_pulsemap(unsorted_map):
    sorted_map = type(unsorted_map)()
    for omkey, pulses in unsorted_map:
        sorted_map[omkey] = type(pulses)(sorted(pulses, key=lambda pulse:pulse.time))
    return sorted_map

def mcpulse_to_recopulse(frame, mapname = "I3MCPulseSeries", outputmap = "I3RecoPulseSeriesMap",bin_width=1*I3Units.nanosecond):
    '''
        A module that does a direct conversion of I3MCPulses to I3RecoPulses.
        It is intended to use on the PMTResponseSimulation output when you
        when one wants to avoid the DOM simulation for some reason (no DOM electronic simulation. ie no launches but
        PMT effects such as saturation is present).
    '''
    from icecube.icetray import I3Units
    recopulsemap = dataclasses.I3RecoPulseSeriesMap()
    mcpulsemap = frame[mapname]
    sorted_mcpulsemap = sort_pulsemap(mcpulsemap)
    for omkey, pulses in sorted_mcpulsemap:
        # In order for some more advanced reconstructions to work
        # properly the pulses need to be coaleced
        if(len(pulses)==0):
            continue
        recopulsemap[omkey] = dataclasses.I3RecoPulseSeries()
        c = pulses[0].charge
        t = pulses[0].time*c
        last_t = pulses[0].time
        for p in pulses[1:]:
            
            #Adding pulse to bin if it is within the bin_width time
            if( (p.time - last_t) < bin_width):
                c += p.charge
                t += p.time*p.charge
                last_t = t/c
            else:
                #creating a new recopulse of the coaleced pulses
                rpulse = dataclasses.I3RecoPulse()
                rpulse.time = t/c
                rpulse.charge = c
                        
                rpulse.width = bin_width
                rpulse.flags = dataclasses.I3RecoPulse.PulseFlags.LC
                if rpulse.charge > 0.25:
                    recopulsemap[omkey].append(rpulse)
                c = p.charge
                t = p.time*p.charge
                last_t = t/c
                
        rpulse = dataclasses.I3RecoPulse()
        #same discriminator threshold cut as IC for now
        # if (c>0.25):
        rpulse.time = t/c
        rpulse.charge = c
        rpulse.width = bin_width
        rpulse.flags = dataclasses.I3RecoPulse.PulseFlags.LC
        if rpulse.charge > 0.25:
            recopulsemap[omkey].append(rpulse)
            
    frame[outputmap] = recopulsemap
    
def mcpe_to_recopulse(frame, mapname = "I3MCPESeriesMap", outputmap = "SplitInIcePulses_UnMasked", bin_width=1*I3Units.nanosecond):
    '''
        A module that does a direct conversion of I3MCPE to I3RecoPulses.
    '''
    from icecube.icetray import I3Units
    recopulsemap = dataclasses.I3RecoPulseSeriesMap()
    mcpemap = frame[mapname]
    sorted_mcpemap = sort_pulsemap(mcpemap) # generic sort pulse/PE in time
    for omkey, pulses in sorted_mcpemap:
        # In order for some more advanced reconstructions to work
        # properly the pulses need to be coaleced
        if(len(pulses)==0):
            continue
        recopulsemap[omkey] = dataclasses.I3RecoPulseSeries()
        c = 1.0 #pulses[0].charge
        t = pulses[0].time #*c
        last_t = pulses[0].time
        for p in pulses[1:]:
            
            #Adding pulse to bin if it is within the bin_width time
            if( (p.time - last_t) < bin_width):
                c += 1.0 #p.charge
                t += p.time #*p.charge
                last_t = t/c
            else:
                #creating a new recopulse of the coaleced pulses
                rpulse = dataclasses.I3RecoPulse()
                rpulse.time = t/c
                rpulse.charge = c
                        
                rpulse.width = bin_width
                rpulse.flags = dataclasses.I3RecoPulse.PulseFlags.LC
                if rpulse.charge > 0.25:
                    recopulsemap[omkey].append(rpulse)
                c = 1.0 #p.charge
                t = p.time #*p.charge
                last_t = t/c
                
        rpulse = dataclasses.I3RecoPulse()
        #same discriminator threshold cut as IC for now
        # if (c>0.25):
        rpulse.time = t/c
        rpulse.charge = c
        rpulse.width = bin_width
        rpulse.flags = dataclasses.I3RecoPulse.PulseFlags.LC
        if rpulse.charge > 0.25:
            recopulsemap[omkey].append(rpulse)
            
    frame[outputmap] = recopulsemap

      
def make_parser():
    parser=OptionParser()

    parser.add_option("-o", "--output", action="store",
      type="string", default="", dest="outfile",
      help="output text file") # location/name of output file


    parser.add_option("-g", "--geom", action="store",
      type="string", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz", dest="geom",
      help="geom?") # input GCD file location
    return parser

    
def create_mask(frame, input="SplitInIcePulses_UnMasked", output="SplitInIcePulses" ):
    newmask = dataclasses.I3RecoPulseSeriesMapMask(frame, input)

    frame[output] = newmask
    
def make_parser():
    parser=OptionParser()

    parser.add_option("-o", "--output", action="store",
      type="string", default="", dest="outfile",
      help="output text file") # location/name of output file


    parser.add_option("-g", "--geom", action="store",
      type="string", default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz", dest="geom",
      help="geom?") # input GCD file location
    return parser


parser = make_parser()
(opts, args) = parser.parse_args()
options = {}
for name in parser.defaults:
    value = getattr(opts,name)
    options[name] = value
    
sub_folder_name = '/mnt/scratch/kochocki/ftp_electrons/' + 'set_' + str(int(float(rngseed - 1)/1000.0)) #1-1000
if not os.path.exists(sub_folder_name ):
    os.mkdir(sub_folder_name )

output_str = sub_folder_name + '/' + options['outfile'] + str(os.getenv('SLURM_ARRAY_TASK_ID'))

    
t = I3Tray()
randomService = phys_services.I3SPRNGRandomService(rngseed, rngnumberofstreams, rngstream)
t.context['I3RandomService'] = randomService
t.context['I3FileStager'] = dataio.get_stagers()
t.AddModule("I3InfiniteSource", "source", prefix = options["geom"], stream = icetray.I3Frame.DAQ) # this should create DAQ frame with proper GCD info
time = dataclasses.I3Time()
time.set_mod_julian_time(int(mjd), 0, 0)
t.AddModule("I3MCEventHeaderGenerator", "time-gen",
        Year=time.utc_year,
        DAQTime=time.utc_daq_time,
        RunNumber=runid,
        IncrementEventID=True)
t.Add(Add_MCTree_Lepton, NEvents=numevents )
t.AddModule(get_mc_truth, 'get_mc_truth', Streams=[icetray.I3Frame.DAQ] ) # adds an object with learning labels, the lepton


photon_kwargs = dict(
        GCDFile = options["geom"],
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
        IceModelLocation=expandvars(os.path.join("$I3_SRC/ice-models/resources/models/ICEMODEL/", "spice_ftp-v2")),
        DisableTilt=False,
        UseGeant4 = False,
        UseCascadeExtension=False,
        CrossoverEnergyEM = 0.1,
        CrossoverEnergyHadron=30.0,
        StopDetectedPhotons=True,
        HoleIceParameterization = expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/angsens/as.flasher_p1_0.35_p2_0"),
        UnshadowedFraction = 0.99,
        PhotonHistoryEntries=0,
        DoNotParallelize=False,
        UseI3PropagatorService=False)
t.Add(clsim.traysegments.I3CLSimMakePhotons, "make_photons", **photon_kwargs)
t.Add(I3CLSimMakeHitsFromPhotons, "make_hits_from_photons",
                     PhotonSeriesName        = "I3PhotonSeriesMap",
                     MCPESeriesName          = "I3MCPESeriesMap",
                     RandomService           = randomService,
                     DOMOversizeFactor       = 1.0,
                     UnshadowedFraction      = 0.99,
                     HoleIceParameterization = expandvars("$I3_SRC/ice-models/resources/models/ANGSENS/angsens/as.flasher_p1_0.35_p2_0"),
                     GCDFile                 = options["geom"],
                     IceModelLocation        = expandvars(os.path.join("$I3_SRC/ice-models/resources/models/ICEMODEL/", "spice_ftp-v2")))
t.Add(mcpe_to_recopulse, "mcpe_to_recopulse",
                     mapname = "I3MCPESeriesMap",
                     outputmap = "SplitInIcePulses_UnMasked" , #"I3RecoPulseSeriesMap"
                     bin_width=1*I3Units.nanosecond, #bin_width=10*I3Units.nanosecond, # For now some baseline value
                     Streams = [icetray.I3Frame.DAQ])
t.AddModule('I3NullSplitter',"null_splitter", InputPulseSeries="SplitInIcePulses_UnMasked",OutputPulseSeriesMask='SplitInIcePulses', SubEventStreamName="StochasticsEventStream_MCPE_to_Reco" )

t.AddModule(lambda frame: frame.Has('SplitInIcePulses') , 'SplitPulses')

# Place all 1000 events into one file
# Place every 1000 (i3/.hdf5) files under one folder
    
t.AddSegment(I3HDFWriter,'I3HDFWriter',output=output_str +'.hdf5',keys= ['LabelsDeepLearning','SplitInIcePulses'], SubEventStreams=['StochasticsEventStream_MCPE_to_Reco'], If =lambda frame: ( len(frame['I3MCPESeriesMap']) > 0)  )

t.Add('I3Writer', Filename= output_str + '.i3')


t.AddModule("TrashCan","trashcan")
t.Execute()
t.Finish()

