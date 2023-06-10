from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from run_recirculation_sdf import start_recirculation
from run_genesis_sdf import *
from run_mergeFiles_sdf import *
from rfp2 import *
import subprocess
from time import sleep
import SDDS
from sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
from scipy.interpolate import interpn, interp2d
import time
import gc
from numpy.random import normal

chirp_jitter = 0.0

nRoundtrips = 0          # number of iteration between ebeam shots
nEbeam_flat_init = 0   # number of ebeam shots
nEbeam_flat = 0
nEbeam_chirp = 30
beam_chirp = 20
pulseLen = 60e-15

ncar = 181
dgrid = 540e-6
w0 =40e-6
xlamds = 1.261043e-10
zsep = 140
c_speed  = 299792458
nslice = 1024
isradi = 1
npadt = (4096 - nslice//isradi)//2
npad1 = (256-ncar)//2
npadx = [int(npad1), int(npad1) + 1]
dt = xlamds*zsep/c_speed
root_dir = '/sdf/group/beamphysics/jytang/genesis/CBXFEL/'
folder_name = 'testMirror'

nametag = 't6'



    

with open(folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
with open(folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")

   

t0 = time.time()
#simulation (change dfl filename)
jobid, sim_name = start_simulation(folder_name = folder_name, dKbyK = 0.03,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = nslice, zsep = zsep,
                                           nametag = nametag,gamma0 = np.around(8000./0.511,3), 
                                           Nf=4, Nt=28, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = pulseLen, sigma = 20e-15, chirp = beam_chirp, Ipeak = 2e3,
                                           xlamds = xlamds,
                                           ipseed=np.random.randint(10000),
                                           prad0 = 2e8)

all_done([jobid])

print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')


    
t0 = time.time()
jobid = start_recirculation(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                 readfilename = root_dir + '/'+folder_name+'/'+sim_name + '.out.dfl' , 
                                 seedfilename = nametag +'_seed_init.dfl',
                                       workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
    
    
    # merge files for each roundtrip on nRoundtrips workers, with larger memory
t0 = time.time()
jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish merging files.')
    

