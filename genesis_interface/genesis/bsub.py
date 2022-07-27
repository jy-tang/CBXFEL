# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import genesis
import subprocess
import os
import glob
from time import sleep
from IPython import display
import re

#-------------------------------------------------
def bsub_script(genesis_binary_path = None, input_file = None, J = None, q = 'beamphys-centos7', W = '1:00', n = '144'):
    """
       Writes a bsub script with slac defaults 
       genesis_binary_dir should point to a file compiled against the correct version of MPI for the q Q  
       Ex: bsub_script(g.genesis_bin,J='jobname',q='beamphys-centos7',W='1:00',n='100')
    """
    
    #workdir = g.path
    
    if J == None: J = 'Genesis_{0}'.format(q)
    if q == 'beamphys-centos7' or q == 'bubble' or q == 'centos7':
        q = 'beamphys-centos7'
        sla = '#BSUB -sla bpc7'
        ptile = 36
        #modules = ['openmpi/3.1.2-gcc-4.8.5']
        if genesis_binary_path == None: gen_bin = os.path.expandvars('$HOME/bin/genesis_BUBBLE')
    elif q == 'beamphysics-mpi' or q == 'bullet':
        q = 'beamphysics-mpi'
        sla = '#BSUB -sla bpmpi'
        ptile = 16
        #modules = ['lsf-openmpi_1.8.1-x86_64']
        if genesis_binary_path == None: gen_bin = os.path.expandvars('$HOME/bin/genesis_BULLET')
    elif q == 'beamphysics' or q == 'oak':
        q = 'beamphysics'
        sla = ''
        ptile = 4
        #modules = ['lsf-openmpi_1.8.1-x86_64']
        if genesis_binary_path == None: gen_bin = os.path.expandvars('$HOME/bin/genesis_OAK')
    else:
        print('Not a known queue')
        return None
    
    if input_file == None:
        input_file = 'genesis.in'
        
    text  = '#!/bin/bash\n'
    text += '#BSUB -J ' + J + '\n' # jobname
    text += '#BSUB -q ' + q + '\n'
    text += '#BSUB -W ' + W + '\n'
    text += '#BSUB -n ' + str(n) + '\n'
    if len(sla) > 0: text += sla + '\n'
    text += '#BSUB -R \"span[ptile=' + str(ptile) + ']\"\n'
    text += '#BSUB -x\n'
    text += '#BSUB -o ' + J + '.job.out\n'
    text += '#BSUB -e ' + J + '.job.err\n'
    text += '#Diagnostics for output file\n'
    text += 'echo -e Started at: `date`\n'
    text += 'echo -e \"Working dir: $LS_SUBCWD\"\n'
    text += 'echo -e \"Master process running on: $HOSTNAME\"\n'
    text += 'echo -e \"Directory is: $PWD\"\n'
    #for module in modules: # doesn't work over ssh
        #text += 'module load ' + module + '\n'
    text += 'mpirun -n ' + str(n) + ' ' + gen_bin + ' ' + input_file + '\n'
    text += '#Finish time\n'
    text += 'echo Finished at: `date`\n'
    
    return text

#-------------------------------------------------
# Input file
#def g_bsub(g,script): 
    #"""
       #Calls gsub using a script. The script is written to file first to allow inspection later (although it should also be written to the file job_%J.out)
    #"""
    #os.chdir(g.path)
    ##write script to directory
    #fname=g.path+'/bsub.sh'
    #with open(fname, 'w') as writer:
        #writer.write(script)
    ##issue call to bsub
    #cmd='bsub'
    #reader=open('bsub.sh')
    #log=[];
    #popen = subprocess.Popen(cmd,stdin=reader,stdout=subprocess.PIPE, universal_newlines=True)  
    #for stdout_line in iter(popen.stdout.readline, ""):
        #log.append(stdout_line)
        ##print(stdout_line)
    #popen.stdout.close()
    #return_code = popen.wait()
    #if return_code:
        #raise subprocess.CalledProcessError(return_code, cmd)
    #return log
    
def g_bsub(g, script, q, verboseQ = False): 
    """
       Calls gsub using a script. The script is written to file first to allow inspection later (although it should also be written to the file job_%J.out)
    """
    pathcmd = ''; ldlibpathcmd = '' # defaults
    if q == 'beamphys-centos7' or q == 'bubble' or q == 'centos7':
        host = 'centos7'
        pathcmd = 'export PATH=/usr/lib64/qt-3.3/bin:/afs/slac/package/lsf/curr/amd64_rhel70/bin:/opt/hpc/gcc-4.8.5/openmpi-3.1.2/fftw-3.3.8/bin:/opt/hpc/gcc-4.8.5/openmpi-3.1.2/parallel-hdf5-1.10.4/bin:/opt/hpc/gcc-4.8.5/openmpi-3.1.2/bin:$PATH:/usr/local/sbin:/usr/sbin'
        ldlibpathcmd = 'export LD_LIBRARY_PATH=/opt/hpc/gcc-4.8.5/openmpi-3.1.2/lib:$LD_LIBRARY_PATH'
    elif q == 'beamphysics-mpi' or q == 'bullet':
        host = 'bullet'
        pathcmd = 'export PATH=/opt/lsf-openmpi/1.8.1/bin:$PATH'
        ldlibpathcmd = 'export LD_LIBRARY_PATH=/opt/lsf-openmpi/1.8.1/lib:$LD_LIBRARY_PATH'
    elif q == 'beamphysics' or q == 'oak':
        host = 'oak'
        pathcmd = 'export PATH=/opt/lsf-openmpi/1.8.1/bin:$PATH'
        ldlibpathcmd = 'export LD_LIBRARY_PATH=/opt/lsf-openmpi/1.8.1/lib:$LD_LIBRARY_PATH'
    else:
        print('Not a known queue or cluster')
        return None
    
    os.chdir(g.path)
    #write script to directory
    fname=g.path+'/bsub.sh'
    with open(fname, 'w') as writer:
        writer.write(script)
    #issue call to bsub
    cmdsep = '; '
    cdcmd = 'cd ' + g.path
    bsubcmd = 'bsub < ' + fname
    cmd = 'ssh ' + host + ' \'' + cmdsep.join([pathcmd,ldlibpathcmd,cdcmd,bsubcmd]) + '\''
    if verboseQ: print(cmd)
    log=[];
    popen = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True) #https://stackoverflow.com/questions/25319277/using-subprocess-to-ssh-and-execute-commands
    #print("got response")
    #response,err = popen.communicate()
    #print(response)
    for stdout_line in iter(popen.stdout.readline, ""):
        log.append(stdout_line)
        #print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return log

def g_bsub_is_finished(g,jid):
    """ waits until gbsub is finished"""
    for p in glob.glob(g.path+"/job*"):
        if re.search('job_{}'.format(jid),p):
            with open(g.path+"/job_{}.out".format(jid)) as fid:
                for line in reversed(fid.readlines()):
                    if re.search('Results reported at',line): return 1;

    return 0

def bjobs():
    """issues bjobs command to the server & returns the output"""
    cmd=['bjobs']
    log=[];
    popen = subprocess.Popen(cmd,stdout=subprocess.PIPE, universal_newlines=True)  
    for stdout_line in iter(popen.stdout.readline, ""):
        log.append(stdout_line)
        #print(stdout_line)
    popen.stdout.close()
    return_code = popen.wait()
    return log

def g_bsub_wait_until_finished(g,jid):
    ii=0
    while g_bsub_is_finished(g,jid)==0:
        sleep(1.5)
        ii=ii+1
        if ii%5==0:
            [print(line) for line in bjobs()]
            display.clear_output(wait=True)
    return None

def g_bsub_wait(g,script): 
    """
       Calls gsub using a script. Waits until finished. Cleans up slice files. Load outputfile. Returns error status (0=run had error)
    """
    glog=g_bsub(g,script)
    jid=re.search('<([0-9]*)',glog[0])
    jid=jid.groups()[0]
    g_bsub_wait_until_finished(g,jid)
    
    for f in glob.glob(g.path+"/*slice*"):
        os.remove(f)
    
    if os.stat("job_{}.err".format(jid)).st_size == 0:
        g.load_output()
        return 1
    else:
        return 0
        
    
