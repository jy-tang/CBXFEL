# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import numpy as np


def write_dpa(pars,path):
    """
   Write pars to dpa. Pars should be in the form given by parse_genesis_dpa.
    """
    dat=pars.flatten()
    dat.tofile(path)
    return None

def write_dfl(dfl,path):
    """
   Write a single z step ("history") to file. dfl should be in the format given by parse_genesis_dfl
    """
        
    if len(np.shape(dfl))!=3:
        print("dfl wrong number of dims. Should be 3. Only prints a single z step.")
        return None
    dat=dfl
    dat=np.moveaxis(dat, [2,1], [1,2]) 
    dat=dat.flatten()
    dat.astype(np.complex).tofile(path)
    return None

def write_beam_file(beamfile,path):
    """
    This function writes a beamfile to path. beamfile should be a dict. A prototype definition would be:
    beamfile=={'ZPOS':zs,'CURPEAK':I, 'ELOSS',eloss} (where zs(meters), I(Amps) and eloss (eV/m) are arrays of the same length). Like all genesis functions zs should be listed tail first & increasing (head greater than tail).
    NTAIL should be set to 0 to match the entered current correctly. If not genesis will linearly interpolate.
    """
    keys=list(beamfile.keys())
    cols=''
    cols=(cols.join([s+' ' for s in keys])).upper()
    if "zs" not in beamfile:
        print("Must have zs")
        return 0
    zs=beamfile["zs"]
    if "zs" not in beamfile:
        print("Must have zs")
    header=(
        "? VERSION=1.0 \n\
? SIZE={:d} \n\
? COLUMNS {:s} \n \n\
".format(len(zs),cols)
    )
    with open(path,'w') as writer:
        writer.write(header)
        for ii in range(len(zs)):
            line=''
            line=line.join(['{:14.8e} '.format(beamfile[k][ii]) for k in keys])
            line=line+'\n'
            writer.write(line)

def write_beam_file(beamfile,path):
    """
    This function writes a beamfile to path. beamfile should be a dict. A prototype definition would be:
    beamfile=={'ZPOS':zs,'CURPEAK':I, 'ELOSS',eloss} (where zs(meters), I(Amps) and eloss (eV/m) are arrays of the same length). Like all genesis functions zs should be listed tail first & increasing (head greater than tail).
    NTAIL should be set to 0 to match the entered current correctly. If not genesis will linearly interpolate.
    """
    keys=list(beamfile.keys())
    cols=''
    cols=(cols.join([s+' ' for s in keys])).upper()
    if "ZPOS" not in beamfile:
        print("Must have ZPOS")
        return 0
    zs=beamfile["ZPOS"]
    header=(
        "? VERSION=1.0 \n\
? SIZE={:d} \n\
? COLUMNS {:s} \n\
".format(len(zs),cols)
    )
    with open(path,'w') as writer:
        writer.write(header)
        for ii in range(len(zs)):
            line=''
            line=line.join(['{:14.8e} '.format(beamfile[k][ii]) for k in keys])
            line=line+'\n'
            writer.write(line)
    return None