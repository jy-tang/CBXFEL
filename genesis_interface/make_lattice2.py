# -*- coding: iso-8859-1 -*-


import numpy as np

# chop to 6 digits since fortran has string limits
def numstr(x, ndecimals=6):
    return str(np.around(x, ndecimals))

# make simplified genesis lattice

#def make_lattice(undKs=[3.51/np.sqrt(2)]*10, und_period=0.030, und_nperiods=110, fodo_length=3.9*2, quad_length=0.3, quad_grad=12.64, latticefilepath='lattice.lat', phaseShifts=None): # LCLS
def make_lattice(undKs=[1.1742,1.1742,[1.1742,1.1771],[1.1771,1.18],[1.18,1.1829],[1.1829,1.1858]]+[1.1735]*10, und_period=0.026, und_nperiods=130, fodo_length=3.9*2, quad_length=0.3, quad_grad=12.64, latticefilepath='lattice.lat', phaseShifts=None): # LCLS2 HXR
#def make_lattice(undKs=[5.48/np.sqrt(2)]*10, und_period=0.039, und_nperiods=87,  fodo_length=3.9*2, quad_length=0.3, quad_grad=12.64, latticefilepath='lattice.lat', phaseShifts=None): # LCLS2 SXR
    # pass a list of undKs to make undulators
    # for each element in undKs, linear taper undulator if element is a list
    # make number of phaseShifts equal to number of undKs

    half_fodo_nper = int(fodo_length / 2. / und_period)
    fill_nper = half_fodo_nper - und_nperiods
    fill_len = fill_nper * und_period
    if fill_nper < 0:
        print('ERROR - make_lattice: undulator is longer than half the FODO length')
    #if fill_nper >= 10:
    #    quad_len_nper = 10
    #else:
    quad_len_nper = max(2, int(quad_length / und_period))
    halfquad_len_nper = quad_len_nper / 2
    quad_grad_scaled = quad_grad * quad_length / und_period / quad_len_nper
        
    if phaseShifts != None:
        if len(undKs) != len(phaseShifts):
            print('ERROR: check that the number of phaseShifts equals the number of undKs')

    lines = ['? VERSION = 1.0', '? UNITLENGTH = '+str(und_period) +' # meters', '']

    for i, K in enumerate(undKs): # Ks must be rms values
        qgrad = (-1)**i * quad_grad_scaled
            
        if len(np.shape(K)) == 0:
            Kend = K
        else:
            Kend = K[-1]

        Nslip = np.ceil(fill_nper / (1 + Kend**2)) # minimize R56 in drift
        AD = np.sqrt((Nslip / fill_nper) * (1 + Kend**2) - 1)
        if phaseShifts != None:
            AD = np.sqrt(AD**2 + (1 + Kend**2) * np.abs(phaseShifts[i]) / (2. * np.pi * fill_nper))
        
        lines += ['QF   ' + numstr(qgrad) + '   ' + str(halfquad_len_nper) + '   0']
        #lines += ['QF   ' + numstr(0.0) + '   ' + str(und_nperiods + quad_len_nper) + '   0']
        lines += ['QF   ' + numstr(0.0) + '   ' + str(half_fodo_nper - quad_len_nper) + '   0']
        lines += ['QF   ' + numstr(-qgrad) + '   ' + str(halfquad_len_nper) + '   0']
        
        lines += ['AD   ' + numstr(AD) + '   ' + str(fill_nper/2) + '   0' ]
        lines += ['AD   ' + numstr(0.0) + '   ' + str(und_nperiods) + '   0' ]
        lines += ['AD   ' + numstr(AD) + '   ' + str(fill_nper/2) + '   0' ]
        
        if len(np.shape(K)) == 0:
            lines += ['AW   ' + numstr(0.0) + '   ' + str(fill_nper/2) + '   0']
            lines += ['AW   ' + numstr(K) + '   ' + str(und_nperiods) + '   0']
            lines += ['AW   ' + numstr(0.0) + '   ' + str(fill_nper/2) + '   0']
        else:
            lines += ['AW   ' + numstr(0.0) + '   ' + str(fill_nper/2) + '   0']
            lines += ['AW   ' + numstr(K[0]) + '   ' + str(1) + '   0']
            for Ki in np.linspace(K[0],K[-1],und_nperiods)[1:]:
                lines += ['AW   ' + numstr(Ki) + '   ' + str(1) + '   ' + str(0)]
            lines += ['AW   ' + numstr(0.0) + '   ' + str(fill_nper/2) + '   0']
        
        
        lines += ['']

    with open(latticefilepath, 'w') as f:
        for l in lines:
            f.write(l+'\n')
