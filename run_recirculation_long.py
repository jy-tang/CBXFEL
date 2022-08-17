from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from cavity_codes.dfl_cbxfel_mpi import *
from genesis_interface.run_genesis_sdf import *
import subprocess
from time import sleep
import SDDS
from genesis_interface.sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
from scipy.interpolate import interpn
import time
import gc
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()


nRoundtrips = 10          # number of iteration between ebeam shots
nEbeam = 2     # number of ebeam shots

ncar = 181
dgrid = 540e-6
w0 =40e-6
xlamds = 1.261043e-10
zsep = 140
c_speed  = 299792458
nslice = 1024
isradi = 1
npadt = (8192 - nslice//isradi)//2
npad1 = (1024-ncar)//2
npadx = [int(npad1), int(npad1) + 1]


folder_name = 'data_long'
#shaped_folder_name = 'flattop_fs'

#record_init_name = "init.txt"
#record_recir_name = "recir.txt"
#record_extract_name = "extract.txt"


for k in range(nEbeam):
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
    
    #   submit genesis job on root node
    if rank == 0:
        if k == 0:   # the first shot starts from noise
            dfl_filename = None
        else:        # others start from recirculated seed
            dfl_filename = 'n' + str(k-1) + 'field_round'+ str(nRoundtrips - 1) + '.dfl'

            # read seed file
            print('start to cut seed file')
            readfilename = './' + folder_name+'/'+dfl_filename
            fld = read_dfl(readfilename, ncar=ncar)

            # cut the central part to match the size of shaped ebeam
            # get the max power position
            power = np.sum(np.abs(fld)**2, axis = (1,2))
            c = np.argmax(power)
            print(c)
            #cut the dfl
            fld = fld[int(c - nslice/2):int(c + nslice/2), :, :]
            print(fld.shape)
            write_dfl(fld, filename = readfilename)

            del fld



        t0 = time.time()
        #simulation (change dfl filename)
        jobid, sim_name = start_simulation(folder_name = folder_name, dKbyK = 0.03,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = nslice, zsep = zsep,
                                           nametag = nametag,gamma0 = np.around(8000./0.511,3), 
                                           Nf=4, Nt=28, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = 60e-15, sigma = 20e-15, chirp = 20, Ipeak = 2e3,
                                           dfl_filename = dfl_filename )

        all_done([jobid])

        print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')
    
    
    # do recirculation on all workers
    recirculate_to_undulator_mpi(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=1.261043e-10,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1,  # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                 readfilename = './' + folder + '/' + sim_name + '.out.dfl' , 
                                       workdir = './' + folder + '/' , saveFilenamePrefix = nametag)
        
    #with open(folder_name+"/" +record_recir_name, "a") as myfile:
    #    myfile.write(" ".join(str(item) for item in return_field_info))
    #    myfile.write("\n")
    #with open(folder_name+"/" +record_init_name, "a") as myfile:
    #    myfile.write(" ".join(str(item) for item in init_field_info))
    #    myfile.write("\n")
    
    #if k%Niter0 == 0:
    #    with open(folder_name+"/" +record_extract_name, "a") as myfile:
    #        myfile.write(" ".join(str(item) for item in  extracted_field_info))
    #        myfile.write("\n")
    gc.collect()
