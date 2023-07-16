from __future__ import print_function
from genesis_interface import genesis
import os, time
import numpy as np
from make_lattice2 import *
from writers import write_beam_file
#from make_lattice import *
import shutil
import subprocess
from time import sleep


def submit_genesis(input_file='HXR.in', shell_script = 'RunGenesis2.sh'):
    cwd = os.getcwd()
    print(cwd)
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

def match_to_FODO(gamma0, emitnx, emitny, L_quad=10*0.026, L_drift=150*0.026, g_quad=14.584615):
    xemit = emitnx/gamma0
    yemit = emitny/gamma0
    
    # calculate matched beta functions

    Lq = L_quad # quad length
    Ld = L_drift # undulator section length
    g = g_quad # quad gradient
    c0 = 299792458.
    mc2 = gamma0*511000.
    f = mc2/g*Lq
    kq = g*c0/mc2
    quadP = np.sqrt(kq)*Lq/2.

    MF = [[np.cos(quadP), np.sin(quadP)/np.sqrt(kq)], [-np.sqrt(kq)*np.sin(quadP), np.cos(quadP)]]
    MD = [[np.cosh(quadP), np.sinh(quadP)/np.sqrt(kq)], [np.sqrt(kq)*np.sinh(quadP), np.cosh(quadP)]]
    ML = [[1, Ld], [0, 1]]

    A = np.dot(MF,np.dot(ML,np.dot(MD,np.dot(MD,np.dot(ML,MF))))) #MF*ML*MD*MD*ML*MF;
    Cphi = A[0,0]
    betaMAX = A[0,1]/np.sqrt(1.-Cphi**2)

    B = np.dot(MD,np.dot(ML,np.dot(MF,np.dot(MF,np.dot(ML,MD))))) #MD*ML*MF*MF*ML*MD;
    Cphi = B[0,0]
    betaMIN = B[0,1]/np.sqrt(1.-Cphi**2)

    xrms_match = np.sqrt(betaMAX * xemit)
    yrms_match = np.sqrt(betaMIN * yemit)
    xprms_match = xemit / xrms_match
    yprms_match = yemit / yrms_match
    
    print('xrms_match = ', xrms_match)
    print('yrms_match = ', yrms_match)
    print('xprms_match = ', xprms_match)
    print('yprms_match = ', yprms_match)
    print('xemit = ', xemit)
    print('yemit = ', yemit)

    return xrms_match, yrms_match

def start_simulation(dKbyK, folder_name, undKs = 1.169, und_period = 0.026, und_nperiods=130, nslice = 1024, zsep = 80, dgrid = 540e-6, ncar = 181, delz = 2, ipseed=0, waitQ = False, verboseQ = True, nametag = '',gamma0 = np.around(8000./0.511,3), Nf = 5, Nt = 27, emitnx = 0.3e-6, emitny = 0.3e-6,pulseLen = 60e-15, sigma = 20e-15, chirp = 20, Ipeak = 2e3,dfl_filename = None, xlamds =1.261043e-10, prad0 = 0, g_quad=14.584615):
    
    root_dir = os.path.realpath(os.path.curdir)
    cwd =root_dir + '/' + folder_name
    os.system('cp  genesis_interface/make_lattice2.py ' + cwd)
    os.system('cp  genesis_interface/genesismp.in ' + cwd)
    os.system('cp  genesis_interface/RunGenesis2.sh ' + cwd)
    
    
    os.chdir(cwd)
    
    params = {
    #'genesis_bin' : os.path.expandvars('$HOME/bin/genesis_BUBBLE'),
    'workdir' : cwd,
    'input_file':cwd+'/genesismp.in',
    'use_tempdir':False,
    }
    
    g = genesis.Genesis(**params)
    
    
    # be mindful of the length of the filename (genesis can only handle ~20 characters)
    sim_name = 'tap'+str(np.around(dKbyK,6))+'_K'+str(np.around(undKs,6))#+'_nt'+str(Nt) + '_nf' + str(Nf)
    
    if len(nametag) > 0:
        sim_name += '_' + nametag
    print(sim_name)
    
    #------------------------Make Lattice----------------------------------------#
    latticefile =  "lattice"+str(int(undKs*1e6))+".dat"
    make_lattice(undKs=[undKs]*40, und_period=und_period, und_nperiods=und_nperiods, fodo_length=3.9*2, quad_length=0.26, quad_grad = g_quad, latticefilepath=latticefile, phaseShifts=None)
    
    
    #------------------------Make Beam--------------------------------------------#
    c_speed = 299792458
    ts = np.linspace(0., pulseLen, 100)
    sigma_z = sigma*c_speed
    zs = ts*c_speed
    #I = np.exp(-(zs - np.mean(zs))**2/2/sigma_z**2)
    #I *= 1/np.max(I)*Ipeak
    I = np.ones((100,))*Ipeak
    gamma = gamma0 + np.linspace(-0.5,0.5,100)*chirp/0.511
    delgam = np.ones((100,))*1.0/0.511
    enx = np.ones((100,))*emitnx
    eny = np.ones((100,))*emitny
    beamfile={'ZPOS':zs,'CURPEAK':I, 'GAMMA0': gamma, 'DELGAM':delgam, 'EMITX':enx, 'EMITY':eny}
    write_beam_file(beamfile,'chirp.beam')
    
    xrms_match, yrms_match = match_to_FODO(gamma0 = gamma0, emitnx = emitnx, emitny = emitny, g_quad = g_quad)
    
    #---------------------Genesis Input-------------------------------------------#
    
    g.input_file = '/'.join(g.input_file.split('/')[:-1] + [sim_name + '.in'])
    g.input['outputfile'] = sim_name + '.out'
    g.input['maginfile'] = latticefile
    g.input['wcoefz(1)'] = 3.9*Nf
    g.input['wcoefz(2)'] = dKbyK # net relative change over whole undulator
    g.input['wcoefz(3)'] =2
    g.input['zstop'] = 3.9*(Nf + Nt)
    #g.input['zstop'] = 0.03*10
    #g.input['ippart'] = 1 # par
    #g.input['idmppar'] = 1 # dpa
   # g.input['xlamds'] = 1.76363e-09
    
    g.input['delz'] = delz# set to 1 for ESASE
    g.input['zsep'] = zsep# set to 1 for ESASE
    #g.input['nslice'] = np.int(1.*g.input['nslice']/g.input['zsep'])
    g.input['ndcut'] = 0  #1000#np.int(g.input['nslice']*0.1)
    g.input['curpeak'] = Ipeak # make sure no random stuff slips in
    g.input['nslice'] = nslice#3865 #0 # have genesis make beam time-slices from the input particle distribution
    #print('curlen = ', g.input['curlen'])
    #print('ntail = ', g.input['ntail'])
    #print('iotail = ', g.input['iotail'])
    #print('isntyp = ', g.input['isntyp'])
    g.input['gamma0'] = gamma0 #15427.59#np.around(7981./0.511,3)
    g.input['delgam'] = 1.0/0.511
    g.input['xlamd'] = und_period
    g.input['xlamds'] = xlamds #1.300e-10 #np.around(g.input['xlamd'] / 2. / g.input['gamma0']**2 * (1. + 1.725**2),9+6) # 4.36514e-10
    g.input['ipseed'] = ipseed
    g.input['npart'] = 4096
    g.input['ishsty'] = 5
    g.input['idmpfld'] = 1
    g.input['ntail'] = 0
    g.input['ncar'] =ncar
    g.input['dgrid'] = dgrid
    g.input['alignradf'] = 1
    
    g.input['prad0'] = prad0
    w0 = 2*xrms_match
    g.input['zrayl'] = np.pi*w0**2/g.input['xlamds']
    
    if dfl_filename:
        g.input['fieldfile'] = dfl_filename
    
    g.input['rxbeam'] = xrms_match
    g.input['rybeam'] = yrms_match
    g.input['emitx'] = emitnx
    g.input['emity'] = emitny
    g.input['alphax'] = 0.
    g.input['alphay'] = 0.

 
     #g.input['fieldfile'] = 'n8.dfl'
    # http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
    g.input['beamfile'] = 'chirp.beam'
    
    g.write_input_file()
    
    
    jobid = submit_genesis(input_file=sim_name+'.in')

    os.chdir(root_dir)
    
    return jobid, sim_name

#jobid, sim_name = start_simulation(folder_name = 'data_long', dKbyK = 0.03,undKs = 1.172,und_period = 0.026,und_nperiods=130, nslice = 1024, zsep = 140,
#                                           nametag = "" ,gamma0 = np.around(8000./0.511,3), 
#                                           Nf=4, Nt=28, emitnx = 0.3e-6, emitny = 0.3e-6,
#                                           pulseLen = 60e-15, sigma = 20e-15, chirp = 20, Ipeak = 2e3,
#                                           dfl_filename =  None)

#all_done([jobid])