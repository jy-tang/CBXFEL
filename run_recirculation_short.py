from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from run_recirculation_sdf import start_recirculation
from run_genesis_sdf import *
from rfp2 import *
import subprocess
from time import sleep
import SDDS
from sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
from scipy.interpolate import interpn
import time
import gc



nRoundtrips = 30          # number of iteration between ebeam shots
nEbeam_flat_init = 10   # number of ebeam shots
nEbeam_flat = 0
nEbeam_chirp = 3
beam_chirp = 20

ncar = 181
dgrid = 540e-6
w0 =40e-6
xlamds = 1.261043e-10
zsep = 200
c_speed  = 299792458
nslice = 710
isradi = 1
npadt = (8192 - nslice//isradi)//2
npad1 = (512-ncar)//2
npadx = [int(npad1), int(npad1) + 1]


root_dir = '/sdf/group/beamphysics/jytang/genesis/CBXFEL/'
folder_name = 'data_short3'
#shaped_folder_name = 'flattop_fs'

#record_init_name = "init.txt"
#record_recir_name = "recir.txt"
#record_extract_name = "extract.txt"

with open('recirc.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
with open('transmit.txt', "w") as myfile:
    myfile.write("Round energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")

for k in range(nEbeam_flat_init + nEbeam_flat*nEbeam_chirp + 1):
    nametag = 'n' + str(k)

   
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
        #cut the dfl
        fld = fld[int(c - nslice/2):int(c + nslice/2), :, :]
        print('fld shape after cutting', fld.shape)
        write_dfl(fld, filename = root_dir+'/' + folder_name + '/'+seed_filename)

        del fld

    #-------------------change beam chirp-----------------------------------------------------
    if k < nEbeam_flat_init or (k - nEbeam_flat_init) %nEbeam_flat != 0:
        print('shot ', str(k), 'flat beam')
        chirp = 0
    else:
        chirp = beam_chirp
        print('shot ', str(k), 'chirped beam')
        
    t0 = time.time()
    #simulation (change dfl filename)
    jobid, sim_name = start_simulation(folder_name = folder_name, dKbyK = 0.006,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = nslice, zsep = zsep,
                                           nametag = nametag,gamma0 = np.around(8000./0.511,3), 
                                           Nf=5, Nt=7, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = 40e-15, sigma = 10e-15, chirp = chirp, Ipeak = 2e3,
                                           ipseed=np.random.randint(10000),
                                           dfl_filename =  seed_filename)

    all_done([jobid])

    print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')
    
    
    
    # do recirculation on all workers
    t0 = time.time()
    jobid = start_recirculation(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=1.261043e-10,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = 12*3.9, l_cavity = 49, w_cavity = 1,  # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                 readfilename = root_dir + '/'+folder_name+'/'+sim_name + '.out.dfl' , 
                                 seedfilename = 'n' + str(k + 1)+'_seed_init.dfl',
                                       workdir = root_dir + '/' + folder_name + '/' , saveFilenamePrefix = nametag)
    
    all_done([jobid])
    print('It takes ', time.time() - t0, ' seconds to finish recirculation.')
        
 
    gc.collect()
