from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from cavity_codes.dfl_cbxfel_mpi import *
from genesis_interface.run_genesis_sdf import *
import subprocess
from time import sleep
import genesis_interface.SDDS
from genesis_interface.sdds2genesis import match_to_FODO,sdds2genesis
import sys, os, csv, copy,shutil
from scipy.interpolate import interpn
import time
import gc

Niter0 = 10          # number of iteration between ebeam shots
Niter_e = 20     # number of ebeam shots

ncar = 256
dgrid = 400e-6
w0 =40e-6
xlamds = 1.261043e-10
zsep = 200
c_speed  = 299792458
nslice = 1024
npadt = 2048
npadx = 512


folder_name = 'S2E/fromZhen/XAMP_0.24_2/'
#shaped_folder_name = 'flattop_fs'




record_init_name = "init.txt"
record_recir_name = "recir.txt"
record_extract_name = "extract.txt"


for k in range(54, Niter0*Niter_e + 1):
    nametag = 'n' + str(k)
    
     #---------------------Make Preparation---------------------------------------------------#
    # make a file to record the energy and size of radiation
    if k ==0:
        with open(folder_name+"/" +record_init_name, "w") as myfile:
            myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
        with open(folder_name+"/" +record_recir_name, "w") as myfile:
            myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
        with open(folder_name+"/" +record_extract_name, "w") as myfile:
            myfile.write("energy/uJ peakpower/GW trms/fs  tfwhm/fs xrms/um  xfwhm/um yrms/um yfwhm/um \n")
            
    
    if k == 0:   # the first shot starts from noise
        dfl_filename = None
    else:        # others start from recirculated seed
        dfl_filename = 'n' + str(k-1) + '.dfl'
        
    if k%Niter0 == 0:         # the shots with electron beam
        zsep = zsep_t
        isradi = isradi_t
        npadt = 1000
        
        extracted_file = './' + folder_name + '/'+nametag+ 'extracted' +'.dfl'   # for the shots with electron beam, record the outcoupled dfl
        
        
    else:
        isradi = 1
        npadt = 0
        zsep = zsep_t * isradi_t
        
        extracted_file = None
        
    if (k > 0) and (k%Niter0 == 0):   # for the shots with electron beam and with seed, cut and interpolate dfl
        
        print("start cutting and interpolating dfl file for shaped beam")
        
        # read seed file
        readfilename = './' + folder_name+'/'+dfl_filename
        fld = read_dfl(readfilename, ncar=ncar)
        print(fld.shape)
        # cut the central part to match the size of shaped ebeam
        a,_,_ = fld.shape
        b = int(nslice/isradi_t)
        
        # get the max power position
        power = np.sum(np.abs(fld)**2, axis = (1,2))
        #print(np.argmax(power))
        c = np.argmax(power)
        print(c,b)
        #cut the dfl
        fld = fld[int(c - b/2):int(c + b/2), :, :]
        print(fld.shape)
        #interpolation to match the slice number of the ebeam
        start = time.time()

        xsize,ysize,zsize = fld.shape

        xisize = nslice

        x = np.arange(xsize)
        y = np.arange(ysize)
        z = np.arange(zsize)
        points = (x, y, z)

        xi = np.linspace(0, xsize -1, xisize)

        interp_points = np.zeros((xisize, ysize, zsize,3))
        interp_points[:,:,:,0] = xi[:, np.newaxis, np.newaxis]
        interp_points[:,:,:,1] = y[np.newaxis, :, np.newaxis]
        interp_points[:,:,:,2] = z[np.newaxis, np.newaxis, :]

        interp_points = np.reshape(interp_points, (xisize * ysize * zsize, 3))

        fld = interpn(points, fld,interp_points,  method='nearest')
        fld = np.reshape(fld, [xisize, ysize, zsize])
        
        print(fld.shape)
        end = time.time()
        print("it takes " + str(end - start)+'s to finish interpolation')
        
        # plot the field after interpolation
        #plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename='./'+folder_name + '/'+dfl_filename+'_interp_xy.png',showPlotQ=False) # plot the imported field
        #energy_uJ,trms, tfwhm, xrms, xfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-2,saveFilename='./'+folder_name+'/'+dfl_filename+'_interp.png',showPlotQ=False) # plot the imported field
        #_,_, _, yrms, yfwhm = plot_fld_slice(fld, dgrid, dt=dt, slice=-1,saveFilename='./'+folder_name+'/'+dfl_filename+'_interp.png',showPlotQ=False) # plot the imported field
        #plot_fld_power(fld, dt=dt, saveFilename=dfl_filename+'_interp.png',showPlotQ=showPlotQ)
        
        #fld_info(fld, dgrid = dgrid, dt=dt)
        
        # write dfl to file for use in the genesis
        write_dfl(fld, filename = readfilename)
        
        del fld
        del interp_points
        
        
    if k%Niter0 == 0:
        #simulation (change dfl filename)
        jobid, sim_name = start_simulation_S2E(dKbyK = 0.05, folder_name = folder_name, undKs = 1.169, und_period = 0.026, und_nperiods=130, ipseed=0, waitQ = False, verboseQ = True, nametag = nametag, gamma0 = np.around(8000./0.511,3), Nf = 3, Nt = 29,  dfl_filename = dfl_filename)
        
        all_done([jobid])
        
        print('Genesis job finished. Start recirculation')
        
        #recirculation from end to start of undulator
        _, init_field_info, return_field_info, time_jitter = recirculate_to_undulator(zsep = zsep, ncar = ncar, dgrid = dgrid, l_undulator=l_undulator, xlamds=1.261043e-10,
                             npadt = npadt, Dpadt = 0, npadx = 50, unpadtQ = False,
                             tjitter_rms = 0, slice_processing_relative_power_threshold = 0,
                                 isradi = isradi, l_cavity = 149, w_cavity = 1,
                                 skipTimeFFTsQ = 1, skipSpaceFFTsQ = 0,
                                 showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                                 readfilename = './'+folder_name+'/'+sim_name + '.out.dfl', writefilename = './'+folder_name+'/'+nametag+'.dfl')
        
        print('Recirculation finished!')
        
       
        
        #outcouple
        _, extracted_field_info = extract_from_cavity(zsep = zsep, ncar = ncar, dgrid = dgrid, l_undulator = l_undulator, xlamds=1.261043e-10,
                        npadx = 50, npadt =npadt, slice_processing_relative_power_threshold = 0,
                        isradi = isradi, l_cavity = 149, w_cavity = 1,unpadtQ = True,
                        skipTimeFFTsQ = 1, skipSpaceFFTsQ = 0, 
                        showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                        readfilename = './'+folder_name+'/'+sim_name + '.out.dfl', writefilename = './'+folder_name+'/'+nametag+ 'extracted' +'.dfl')
        
        
    else:
        #recirculation roundtrip(from start to start)
        _, init_field_info, return_field_info, time_jitter = recirculate_roundtrip(zsep=zsep, ncar=ncar, dgrid=dgrid, l_undulator=l_undulator, xlamds=1.261043e-10,
                             npadt = npadt, Dpadt = 0, npadx = 50, unpadtQ = False,
                             tjitter_rms = 0, slice_processing_relative_power_threshold = 0,
                                 isradi = isradi, l_cavity = 149, w_cavity = 1,
                                 skipTimeFFTsQ = 1, skipSpaceFFTsQ = 0,
                                 showPlotQ = 0, savePlotQ = 0, verbosity = 1,
                                 readfilename =  './'+folder_name+'/'+dfl_filename, writefilename = './'+folder_name+'/'+nametag +'.dfl')
        
    
    with open(folder_name+"/" +record_recir_name, "a") as myfile:
        myfile.write(" ".join(str(item) for item in return_field_info))
        myfile.write("\n")
    with open(folder_name+"/" +record_init_name, "a") as myfile:
        myfile.write(" ".join(str(item) for item in init_field_info))
        myfile.write("\n")
    
    if k%Niter0 == 0:
        with open(folder_name+"/" +record_extract_name, "a") as myfile:
            myfile.write(" ".join(str(item) for item in  extracted_field_info))
            myfile.write("\n")
    gc.collect()
