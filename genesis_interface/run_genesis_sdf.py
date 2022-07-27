from __future__ import print_function
from genesis import genesis, parsers, lattice, analysis, lattice_operations, bsub
import os, time
import numpy as np
from make_lattice2 import *
#from make_lattice import *
import shutil
import subprocess
from time import sleep


def submit_genesis(input_file='HXR.in', shell_script = 'RunGenesis2.sh'):
    cmd = 'sbatch '+ shell_script + ' ' + input_file
    print(cmd)
    x = subprocess.check_output(cmd.split())
    y = x.split()
    print(x)
    return y[-1]

def all_done(jid):
    flag = [False for i in range(len(jid))]
    all_done_flag = False
    while not all_done_flag:
        sleep(30)
        count = 0
        for id in jid:
            ret2=subprocess.getoutput("squeue -u jytang")
            if ret2.count(str(int(id))) < 1:
                flag[count] = True
            count +=1
        all_done_flag = all(flag)
        print("job "  + str(jid[0]) + " is running")
    print('all done!')



def start_simulation(dKbyK, folder_name, undKs = 1.169, und_period = 0.026, und_nperiods=130, ipseed=0, waitQ = False, verboseQ = True, nametag = '',
                       gamma0 = np.around(8000./0.511,3), Nf = 5, Nt = 27):
    
    root_dir = os.path.realpath(os.path.curdir)
    cwd =root_dir + '/' + folder_name
    os.system('cp  ./make_lattice.py ' + cwd)
    os.system('cp  ./genesismp_s2e.in ' + cwd)
    os.system('cp ./RunGenesis2.sh ' + cwd)
    
    
    os.chdir(cwd)
    
    params = {
    #'genesis_bin' : os.path.expandvars('$HOME/bin/genesis_BUBBLE'),
    'workdir' : cwd,
    'input_file':cwd+'/genesismp_s2e.in',
    'use_tempdir':False,
    }
    
    g = genesis.Genesis(**params)
    
    
    # be mindful of the length of the filename (genesis can only handle ~20 characters)
    sim_name = 'gen_tap'+str(np.around(dKbyK,6))+'_K'+str(np.around(undKs,6))+'_n'+str(Nf)
    
    if len(nametag) > 0:
        sim_name += '_' + nametag
    print(sim_name)
    
    latticefile =  "lattice"+str(int(undKs*1e6))+".dat"
    make_lattice(undKs=[undKs]*40,latticefilepath=latticefile,und_period=und_period, und_nperiods = und_nperiods)
    #Nf = 5
    #Nt = 27
    
    
    g.input_file = '/'.join(g.input_file.split('/')[:-1] + [sim_name + '.in'])
    g.input['outputfile'] = sim_name + '.out'
    g.input['maginfile'] = latticefile
    g.input['wcoefz(1)'] = 3.9*Nf
    g.input['wcoefz(2)'] = dKbyK # net relative change over whole undulator
    g.input['wcoefz(3)'] =1
    g.input['zstop'] = 3.9*(Nf + Nt)
    #g.input['zstop'] = 0.03*10
    #g.input['ippart'] = 1 # par
    #g.input['idmppar'] = 1 # dpa
   # g.input['xlamds'] = 1.76363e-09
    
    g.input['delz'] = 1# set to 1 for ESASE
    g.input['zsep'] = 1# set to 1 for ESASE
    #g.input['nslice'] = np.int(1.*g.input['nslice']/g.input['zsep'])
    g.input['ndcut'] = 0  #1000#np.int(g.input['nslice']*0.1)
    g.input['curpeak'] = 1. # make sure no random stuff slips in
    g.input['nslice'] = 0#3865 #0 # have genesis make beam time-slices from the input particle distribution
    #print('curlen = ', g.input['curlen'])
    #print('ntail = ', g.input['ntail'])
    #print('iotail = ', g.input['iotail'])
    #print('isntyp = ', g.input['isntyp'])
    g.input['gamma0'] = gamma0 #15427.59#np.around(7981./0.511,3)
    g.input['delgam'] = 1.0/0.511
    g.input['xlamd'] = und_period
    g.input['xlamds'] = 1.261043e-10 #1.300e-10 #np.around(g.input['xlamd'] / 2. / g.input['gamma0']**2 * (1. + 1.725**2),9+6) # 4.36514e-10
    g.input['ipseed'] = ipseed
    g.input['npart'] = 4096
    g.input['ishsty'] = 5
    g.input['idmpfld'] = 0
    g.input['ncar'] =181
    g.input['dgrid'] = 540e-6
    #g.input['alignradf'] = 1
    
    g.input['prad0'] = 2e9
    w0 = 15e-6
    g.input['zrayl'] = np.pi*w0**2/g.input['xlamds']
 
     #g.input['fieldfile'] = 'n8.dfl'
    # http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
    g.input['distfile'] = 'HXRSTART_matchproj.dist'
    
    g.write_input_file()
    
    
    
    jobid = submit_genesis(input_file=sim_name+'.in')

    os.chdir(root_dir)
    
    return jobid, sim_name

def start_simulation_as(K1, taper1, K2, taper2, Nf1, Nf2, Nt1, Nt2, folder_name, zsep =40, nslice = 5350, ncar = 181, dgrid = 540e-6, dfl_filename = None,  ipseed=0, nametag = ''):
    
    root_dir = os.path.realpath(os.path.curdir)
    cwd =root_dir + '/' + folder_name
    os.system('cp  ./make_lattice.py ' + cwd)
    os.system('cp  ./genesismp_s2e.in ' + cwd)
    os.system('cp ./RunGenesis2.sh ' + cwd)
    
    
    os.chdir(cwd)
    
    params = {
    #'genesis_bin' : os.path.expandvars('$HOME/bin/genesis_BUBBLE'),
    'workdir' : cwd,
    'input_file':cwd+'/genesismp_s2e.in',
    'use_tempdir':False,
    }
    
    g = genesis.Genesis(**params)
    
    
    # be mindful of the length of the filename (genesis can only handle ~20 characters)
    sim_name = 'gen_tap'+str(np.around(taper2,6))+'_K'+str(np.around(K2,6))
    
    if len(nametag) > 0:
        sim_name += '_' + nametag
    print(sim_name)
    
    #latticefile =  "lattice"+str(int(K1*1e6))+".dat"
    #make_lattice(undKs=[undKs]*40,latticefilepath=latticefile,und_period=und_period, und_nperiods = und_nperiods)
    #Nf = 5
    #Nt = 27
    
    latticefile = "lattice.lat"
    K1end = K1*(1+taper1)
    temp1 = np.linspace(K1,K1end,Nt1+1)
    undK_list = [K1]*Nf1
    for i in range(1,Nt1+1):
        undK_list += [[temp1[i-1],temp1[i]]]
    undK_list += [K2]*(Nf2+Nt2)   
    make_lattice(undKs = undK_list,latticefilepath = latticefile)
    
    
    
    g.input_file = '/'.join(g.input_file.split('/')[:-1] + [sim_name + '.in'])
    g.input['outputfile'] = sim_name + '.out'
    g.input['maginfile'] = latticefile
    g.input['wcoefz(1)'] = 3.9*(Nf1+Nt1+Nf2)
    g.input['wcoefz(2)'] = taper2  # net relative change over whole undulator
    g.input['wcoefz(3)'] =2
    g.input['zstop'] = 3.9*(Nf1+Nt1+Nf2+Nt2)
    #g.input['zstop'] = 0.03*10
    #g.input['ippart'] = 1 # par
    #g.input['idmppar'] = 1 # dpa
   # g.input['xlamds'] = 1.76363e-09
    
    g.input['delz'] = 1# set to 1 for ESASE
    g.input['zsep'] = zsep# set to 1 for ESASE
    #g.input['nslice'] = np.int(1.*g.input['nslice']/g.input['zsep'])
    g.input['ndcut'] = 0  #1000#np.int(g.input['nslice']*0.1)
    g.input['curpeak'] = 1. # make sure no random stuff slips in
    g.input['nslice'] = nslice #3865 #0 # have genesis make beam time-slices from the input particle distribution
    #print('curlen = ', g.input['curlen'])
    #print('ntail = ', g.input['ntail'])
    #print('iotail = ', g.input['iotail'])
    #print('isntyp = ', g.input['isntyp'])
    g.input['gamma0'] = 15634.634 #15427.59#np.around(7981./0.511,3)
    g.input['delgam'] = 1.0/0.511
    g.input['xlamd'] = 0.026
    g.input['xlamds'] = 1.261043e-10 #1.300e-10 #np.around(g.input['xlamd'] / 2. / g.input['gamma0']**2 * (1. + 1.725**2),9+6) # 4.36514e-10
    g.input['ipseed'] = ipseed
    g.input['npart'] = 4096
    g.input['ishsty'] = 5
    g.input['idmpfld'] = 1
    g.input['ncar'] =ncar
    g.input['dgrid'] = dgrid
    g.input['alignradf'] = 1
    
    g.input['prad0'] = 0 #2e9
    w0 = 15e-6
    g.input['zrayl'] = np.pi*w0**2/g.input['xlamds']
    
    if dfl_filename:
        g.input['fieldfile'] = dfl_filename
     #g.input['fieldfile'] = 'n8.dfl'
    # http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
    g.input['distfile'] = 'HXRSTART_matchproj.dist'
    
    g.write_input_file()
    
    
    
    jobid = submit_genesis(input_file=sim_name+'.in')

    os.chdir(root_dir)
    
    return jobid, sim_name
