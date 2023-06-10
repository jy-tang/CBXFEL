from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from run_recirculation_sdf import start_recirculation_4lens
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



nRoundtrips = 9         # number of iteration between ebeam shots
nEbeam_flat_init = 16   # number of ebeam shots
nEbeam_flat = 5
nEbeam_chirp = 2
beam_chirp = 5
pulseLen = 50e-15
sigma = 50e-15
Ipeak = 2e3


w0 =40e-6
xlamds = 6.30524765e-11
zsep = 200
ncar = 181
dgrid = 360.e-6
delz = 2

undK = 0.4336
taper = 0.01
Nf = 25
Nt = 7

c_speed  = 299792458
nslice = 1500
isradi = 1
npadt = (8192 - nslice//isradi)//2
npad1 = (512-ncar)//2
npadx = [int(npad1), int(npad1) + 1]
dt = xlamds*zsep/c_speed


root_dir = '/sdf/group/beamphysics/jytang/genesis/CBXFEL/'
folder_name = 'data_20keV_taper4'



if nEbeam_flat_init > 0 and nEbeam_flat > 0:
    Nshot_total = nEbeam_flat_init + nEbeam_flat*nEbeam_chirp + 1
else:
    Nshot_total = nEbeam_chirp + 1


    
for k in range(Nshot_total):
    nametag = 'n' + str(k)
    with open(folder_name + '/'+nametag+'_recirc.txt', "w") as myfile:
        myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
    with open(folder_name + '/'+nametag+'_transmit.txt', "w") as myfile:
        myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")

   
   
    
     #---------------------Make Preparation---------------------------------------------------#
    # make a file to record the energy and size of radiation
    #if k ==0:
    #    with open(folder_name+"/" +record_init_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    #    with open(folder_name+"/" +record_recir_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    #    with open(folder_name+"/" +record_extract_name, "w") as myfile:
    #        myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
    
    #   submit genesis job 
     
    #-------------------Prepare Seed Beam----------------------------------------------------#
    if k == 0:   # the first shot starts from noise
        dfl_filename = None
        seed_filename = None
    else:        # others start from recirculated seed
        dfl_filename = nametag+'_seed_init.dfl'  # the seed file without cutting from the last electron shots
        seed_filename =  nametag+'_seed.dfl'

        # read seed file
        print('start to cut seed file')
        readfilename = root_dir + '/' + folder_name+'/'+dfl_filename
        fld = read_dfl(readfilename, ncar=ncar)
        print(fld.shape)
        print("finished read dfl")
        # cut the central part to match the size of shaped ebeam
        # get the max power position
        power = np.sum(np.abs(fld)**2, axis = (1,2))
        c = np.argmax(power)
        shift = int(len(power)//2 - c)
        fld = np.roll(fld, shift, axis = 0)
        
        power = np.sum(np.abs(fld)**2, axis = (1,2))
        c = np.argmax(power)
        fld = fld[int(c - nslice/2):int(c + nslice/2), :, :]
        print('fld shape after cutting', fld.shape)
        write_dfl(fld, filename = root_dir+'/' + folder_name + '/'+seed_filename)

        del fld

    #-------------------change beam chirp-----------------------------------------------------
    if nEbeam_flat_init > 0 and  nEbeam_flat > 0 and (k < nEbeam_flat_init or (k - nEbeam_flat_init) %nEbeam_flat != 0):
        print('shot ', str(k), 'flat beam')
        chirp = 0
    else:
        chirp = beam_chirp
        
    t0 = time.time()
    #simulation (change dfl filename)
    jobid, sim_name = start_simulation(folder_name = folder_name, dKbyK = taper,undKs = undK,und_period = 0.026,und_nperiods=130, nslice = nslice, zsep = zsep,ncar = ncar, dgrid = dgrid, delz = delz, g_quad=5.0,
                                           nametag = nametag,gamma0 = np.around(8000./0.511,3), 
                                           Nf=Nf, Nt=Nt, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = pulseLen, sigma = sigma, chirp = chirp, Ipeak = Ipeak,
                                           xlamds = xlamds,
                                           ipseed=np.random.randint(10000),
                                           dfl_filename =  seed_filename)

    all_done([jobid])

    print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')
    
    
    
    # do recirculation on all workers
    t0 = time.time()
    jobid = start_recirculation_4lens(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                 readfilename = root_dir + '/'+folder_name+'/'+sim_name + '.out.dfl' , 
                                 seedfilename = 'n' + str(k + 1)+'_seed_init.dfl',
                                       workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
    all_done([jobid])
    print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
    
    
    # merge files for each roundtrip on nRoundtrips workers, with larger memory
    t0 = time.time()
    jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = root_dir + '/' + folder_name + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    
    all_done([jobid])
    print('It takes ', time.time() - t0, ' seconds to finish merging files.')
    gc.collect()
