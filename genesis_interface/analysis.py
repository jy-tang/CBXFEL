# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import *
from genesis_interface import parsers

#speed up fawley algorithim
from numba import jit 
from math import sin

def load_slices(fname):

    gout = parsers.parse_genesis_out(fname)
    slices = gout['slice_data']
    zsep = gout['input_parameters']['zsep']
    xlamds = gout['input_parameters']['xlamds'] 
    Nslice = len(slices)
   
    return slices, zsep, Nslice, xlamds

def power(slices):

    power = np.asarray([s['data']['power'][:] for s in slices])
    
    return power

def espread(slices):

    espread = np.asarray([s['data']['e-spread'][:] for s in slices])

    return espread

def bunching(slices):

    bunching = np.asarray([s['data']['bunching'][:] for s in slices])

    return bunching

def current_profile(slices):
    current = np.asarray([s['current'] for s in slices])

    return current

def power_spectrum(slices):

    Z0 = 120 * np.pi
    power = np.asarray([s['data']['p_mid'][-1] for s in slices])
    phi_mid = np.asarray([s['data']['phi_mid'][-1] for s in slices])
    field = np.sqrt(power) * np.exp(1j*phi_mid)
    power_fft = np.abs(np.fft.fftshift(np.fft.fft(field)))**2

    return power_fft


def freq_domain_eV(zsep,Nslice,xlamds):

    #constants
    hbar = 6.582e-16 #in eV
    c = 2.997925e8

    #omega of the radiation in eV
    omega = hbar * 2.0 * np.pi / (xlamds/c);
    df = hbar * 2.0 * np.pi/Nslice/zsep/(xlamds/c);

    freq = np.linspace(omega - Nslice/2 * df, omega + Nslice/2 * df,Nslice)

    return freq


def gaussian(x, *p):

    A, mu, sigma, bg = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + bg

def FWHM(power_fft,omega):

    power_fft = power_fft / np.max(power_fft)
    peak_pos = np.argmax(power_fft)
    p0 = [1.0, omega[peak_pos], 0.15, 1e-4]
    window = 10
    coeff, var_matrix = curve_fit(gaussian, omega[peak_pos-window:peak_pos+window], power_fft[peak_pos-window:peak_pos+window], p0=p0)
    FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0)) * coeff[2]

    print ('Fitted mean = ', coeff[1])
    print ('Fitted standard deviation = ', coeff[2])
    print ('Fitted FWHM = ', FWHM)

    return coeff, FWHM

def calculate_JJ(K):

    J_arg = K**2 / (1.0 + K**2) / 2.0
    JJ = j0(J_arg) - j1(J_arg)

    return JJ

def calculate_gamma_res(g, K):

    lambdau = g.input['xlamd']
    lambdas = g.input['xlamds']
    gamma_r2 = (lambdau/(2.0 * lambdas)) * (1.0 + K**2 )
    gamma_res = np.sqrt(gamma_r2)

    return gamma_res

def calculate_AD(K, Ngap):

    NP = np.ceil(Ngap/(1 + K**2))
    AD = np.sqrt( (1 + K**2 - Ngap/NP) / (Ngap/NP)  )

    return AD

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

def resonant_wavelength(K,ku,gamma):
    return (np.pi/(ku)/gamma**2)*(1+K**2)

def g_resonant_wavelength(g,K=None,ku=None,gamma=None):
    if K is None: K =g.input['aw0']
    if ku is None: ku =np.pi*2/g.input['xlamd']
    if gamma is None: gamma = g.input['gamma0']
    return resonant_wavelength(K,ku,gamma)

def MingXie(Kgen,ku,gamma,rel_e_spread,I,beam_size,normemittance,wavelength,iwityp):
    IA=17.045e3;
    fc=(jv(0,Kgen**2/2/(1+Kgen**2))-jv(1,Kgen**2/2/(1+Kgen**2))) if iwityp==0 else 1.0
    rho=1/gamma*((Kgen*fc/(4*ku*beam_size))**2*I/IA)**(1/3)
    Lg1d=1/2/ku/rho/np.sqrt(3)
    
    Lr=4*np.pi*beam_size**2/wavelength
    nd=Lg1d/Lr
    
    geoemittance=normemittance/gamma
    beta=beam_size**2/geoemittance
    ne=(Lg1d/beta)*(4*np.pi*geoemittance/wavelength)
    ng=2*(Lg1d*ku)*(rel_e_spread)
    
    a=[0,0.45,0.57,0.55,1.6,3,2,0.35,2.9,2.4,51,0.95,3,5.4,0.7,1.9,1140,2.2,2.9,3.2]
    n=(a[1]*(nd**a[2]) + a[3]*(ne**a[4]) + a[5]*(ng**a[6]) +
       a[7]*(ne**a[8])*(ng**a[9]) + a[10]*(nd**a[11])*(ng**a[12]) +a[13]*(nd**a[14])*(ne**a[15]) +
       a[16]*(nd**a[17])*(ne**a[18])*(ng**a[19]))
    Lg=Lg1d*(n+1)
    Pbeam=gamma*0.511e-3*I*1e-3*1e12 #from Pbeam[TW]=E0[GeV]I[kA]
    Pnoise=gamma*0.511e6*3e8*1.6e-19*rho**2/wavelength
    Psat=1.6*rho*(Lg1d/Lg)**2*Pbeam
    return {"rho":rho,"Lg":Lg,"Lg1d":Lg1d,"Pbeam":Pbeam,"Pnoise":Pnoise,"Psat":Psat}

def g_MingXie(g,Kgen=None,ku=None,gamma=None,rel_e_spread=None,I=None,beam_size=None,normemittance=None,wavelength=None,iwityp=None):
    if Kgen is None: Kgen=g.input['aw0']
    if ku is None: ku=2*np.pi/g.input['xlamd']
    if gamma is None: gamma=g.input['gamma0']
    if rel_e_spread is None: rel_e_spread=g.input['delgam']/gamma
    if I is None: I=g.input['curpeak']
    if beam_size is None: beam_size=np.sqrt(g.input['rxbeam']**2+g.input['rybeam']**2)
    if normemittance is None: normemittance=np.sqrt(g.input['emitx']**2+g.input['emity']**2)
    if wavelength is None: wavelength=g.input['xlamds']
    if iwityp is None: iwityp=g.input['iwityp']
    return MingXie(Kgen,ku,gamma,rel_e_spread,I,beam_size,normemittance,wavelength,iwityp)




#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#fld analysis
#The field is normalized so that the square sum of the real and imaginary part yields the field intensity with respect to a unit area
def dfl_to_I(dfl,ncar,dgrid):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) ncar, and dgrid, and returns the intensity (in SI units, W/m^2).
    """
    #dgrid is in m, the whole grid goes from [-dgrid,dgrid] in both dimensions.
    #NCAR must be an odd number to cover the undulator axis with one grid point
    #in genesis source code dxy=xkw0*2.d0*dgrid/float(ncar-1) & he uses dxy to get coordinates like xcr=dxy*float(ix-1)/xkw0-dgrid
    area=(dgrid*2/(ncar-1))**2
    return np.abs(dfl)**2/area

def g_dfl_to_I(g,dfl):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) and gets (ncar, dgrid) from g. It returns the intensity (in SI units, W/m^2).
    """
    return dfl_to_I(dfl,g.input['ncar'],g.input['dgrid'])

def dfl_to_E(dfl,ncar,dgrid):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) ncar, and dgrid, and returns the electric field (in SI units, V/m)
    """
    #compare to intensity
    area=(dgrid*2/(ncar-1))**2
    return np.sqrt(2*376.7)*dfl/np.sqrt(area)

def g_dfl_to_E(g,dfl):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) and gets (ncar, dgrid) from g. It returns the the electric field (in SI units, V/m)
    """
    return dfl_to_E(dfl,g.input['ncar'],g.input['dgrid'])

def E_to_dfl(E,ncar,dgrid):
    """
    This function takes an E with a format from parsers.parse_genesis_dfl and returns the units of the dfl file (in genesis units, sqrt(W/grid)). It is the inverse of analysis.dfl_to_E
    """
    #compare to intensity
    area=(dgrid*2/(ncar-1))**2
    return 1/np.sqrt(2*376.7)*E*np.sqrt(area)

def g_E_to_dfl(g,E):
    """
    This function takes an E with a format from parsers.parse_genesis_dfl and gets (ncar, dgrid) from g. It returns the units of the dfl file (in genesis units, sqrt(W/grid)). It is the inverse of analysis.dfl_to_E.
    """
    return E_to_dfl(E,g.input['ncar'],g.input['dgrid'])

def g_fft_field(g,dfl):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) and a genesis object and returns a list and an array (w, E(w)).
    """
    nz=g.input['nslice']
    ts=(np.asarray(range(nz))-nz/2.0)*g.input['xlamds']*g.input['zsep']/3e8
    ws=2*np.pi*np.fft.fftfreq(len(ts),np.median(np.diff(ts)))
    Et=dfl_to_E(dfl,g.input['ncar'],g.input['dgrid'])
    return ws,np.fft.fft(Et,axis=0)
def g_ifft_dfl(g,Ew):
    """
    This function inverts fft_field. It returns (t, E(t)).
    """
    nz=g.input['nslice']
    ts=(np.asarray(range(nz))-nz/2.0)*g.input['xlamds']*g.input['zsep']/3e8
    Et=np.fft.ifft(Ew,axis=0)
    return ts,E_to_dfl(Et,g.input['ncar'],g.input['dgrid'])
def g_fft_intensity(g,dfl):
    """
    This function takes a dfl (from parsers.parse_genesis_dfl) and a genesis object and returns a list and an array (w, I(w)).
    """
    nz=g.input['nslice']
    ts=(np.asarray(range(nz))-nz/2.0)*g.input['xlamds']*g.input['zsep']/3e8
    ws=2*np.pi*np.fft.fftfreq(len(ts),np.median(np.diff(ts)))
    Et=dfl_to_E(dfl,g.input['ncar'],g.input['dgrid'])
    Ew=np.fft.fft(Et,axis=0)
    return ws,1/(2*376.7)*np.abs(Ew)**2

def dfl_energy(g,dfl):
    """calculate the energy in a dfl (format from parsers.parse_genesis_dfl). Returned in J"""
    nz=g.input['nslice']
    ts=(np.asarray(range(nz))-nz/2.0)*g.input['xlamds']*g.input['zsep']/3e8
    return np.trapz(np.trapz(np.trapz(np.abs(dfl[:,:,:])**2,ts,axis=0),axis=0),axis=0)
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#particle analysis

def g_mirror_pars(g,pars,NBINS=None):
    """
    Takes pars like from parsers.parse_genesis_dpa. It returns a "mirrored" set of particles to make NBINS beamlets. This is the fawley tehcnique for setting bunching to 0. Note that NBINS/2 should be bigger than the biggest harmonic you care about. Also not that this function increases the number of particles by a factor of NBINS. 
    This function sets the meanphase to 0 (dpa phase keeps track of absolute distance in genesis).
    
    This function returns: parout (the new par array), NBINS, and the new number of particles
    """
    if NBINS==None:
        NBINS=g.input['NBINS'] 
    parout=np.tile(pars,NBINS)
    if g.input['itdp']==0:
        ns=1
    else:
        ns=g.input['nslice']
    phases=np.linspace(0,(2*np.pi)-(2*np.pi)/(NBINS),NBINS)
    mirrorphases=np.concatenate([np.ones([ns,g.input['npart']])*p for p in phases],axis=1)
    meanphase=np.mean(parout[:,1,:])
    parout[:,1,:]=parout[:,1,:]+mirrorphases-meanphase
    return parout, NBINS, g.input['npart']*NBINS


@jit(nopython=True)
def fawley(par,I,efac,NBINS,npart,pl):
    """
    This function is copied from genesis loadbeam.f. It is meant to be called by the helper g_fawley. It contains all inner loops in a numb @jit(nopython=True) decorator which compiles them in "just in time" so that they are fast.
    """
    mpart=int(npart/NBINS)
    for cc in range(len(I)):
        enumber=I[cc]*efac
        if enumber<10: # from genesis
            enumber=10
        pl=pl*0.0
        for ih in range(int(np.floor((NBINS-1)/2))): 
            for ii in range(mpart):
                bunch_phase=2*np.pi*np.random.rand()
                bunch_amp=np.sqrt(-np.log(np.random.rand())/enumber)*2/(ih+1)
                for j in range(NBINS):
                    jj=j*mpart+ii
                    pl[jj]=pl[jj]-bunch_amp*np.sin((ih+1)*par[cc,1,jj]+bunch_phase)
        par[cc,1,:]=par[cc,1,:]+pl
    return par
def g_fawley(g,par,I=[None],seed=1):
    """
    This expects a par array from analysis.mirror_pars which has beamlets of 0 bunching. It adds constructs the beamlet phase to have the right bunching. The current must be provided or be in g.output. The seed can be changed to reset numpy's random number generator.
    """
    #NBINS macroparticles make a "beamlet" with a net phasor of bunch_phase and bunch_amp
    #There are mpart "beamlets"
    #To make the "beamlet" have the right bunch_phase/amp we sum over a beamlet
    np.random.seed(seed)
    if len(I)==1:
        I=np.asarray([g.output['slice_data'][ii]['current'] for ii in range(g.input['nslice'])])
    NBINS=g.input['nbins']
    npart=g.input['npart']
    mpart=int(npart/NBINS)
    efac=g.input['xlamds']*g.input['zsep']/3e8/mpart/1.609e-19
    pl=np.zeros([int(npart)])
    fawpar=par.copy()
    par=fawley(fawpar,I,efac,NBINS,npart,pl)
    return par

@jit(nopython=True)
def bunching(pars):
    """
    This function calculated the bunching of a pars array (e.g. from parsers.parse_genesis_dpa). It returns the bunching. The Numba version is marginally faster.
    """
    return np.sum(np.exp(1j*pars[:,1,:]),axis=1)/np.shape(pars)[2]