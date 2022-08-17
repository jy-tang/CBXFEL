from __future__ import print_function
from genesis_interface import genesis
import os, time
import numpy as np
from make_lattice2 import *
from genesis_interface.writers import write_beam_file
#from make_lattice import *
import shutil
import subprocess
from time import sleep
from run_genesis_sdf import *



folder_name = 'data_long'
#shaped_folder_name = 'flattop_fs'

#record_init_name = "init.txt"
#record_recir_name = "recir.txt"
#record_extract_name = "extract.txt"



        
t0 = time.time()
        #simulation (change dfl filename)

jobid, sim_name = start_simulation(folder_name = 'data_long', dKbyK = 0.03,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = 1024, zsep = 140,
                                           nametag = "" ,gamma0 = np.around(8000./0.511,3), 
                                           Nf=4, Nt=28, emitnx = 0.3e-6, emitny = 0.3e-6,
                                           pulseLen = 60e-15, sigma = 20e-15, chirp = 20, Ipeak = 2e3,
                                           dfl_filename =  None)

all_done([jobid])
print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')
    

    
    