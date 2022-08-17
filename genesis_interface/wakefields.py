# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import numpy as np


def rw_wakefield(s,r,s0,tau=None,rf=None):
    """
    E = rw_wakefield(s,r,s0[,tau,rf]);

Function to calculate the resistive wall wakefield (Green's function)
for a bunch which is short or long compared to the characteristic
length for a cylindrical or parallel plate vacuum chamber.
  
Uses the Green's function from Karl Bane's 2004 AC-wake paper for the
case where "tau" is given (AC-wake) and uses the wake from Frank
Zimmermann's and Tor Raubenheimer's NLC-note for the tau=0 (DC) case.

   Inputs:	s:		(vector) Axial position (s>=0) [m]
            r:		Radius of beam chamber [m]
            s0:		Characteristic length (2r^2/(Z0*sigC))^(1/3) [m]
            tau:	(Optional, DEF=none) Relaxation time [sec] - if not given
                  we default to DC-conductivity
            rf:		(Optional, DEF=1) rf=0 is round pipe, and rf=1 is flat (parallel plates)
   Outputs:	E:		Green's function [V/C/m]
    """

    if any(s<0):
        print('s should be >= 0')
        return None

    s_s0 = s/s0;
    
    Z0 = 120*np.pi; 
    c  = 2.99792458E8;
    sig = 2*r**2/Z0/s0**3;

    if rf==None:
        rf = 0;

    if tau!=None:
        Gamma = c*tau/s0;
        if Gamma<=2.5:
            krs0c = np.asarray([
                [1.81620118662482,   1.29832152141677],
                [0.29540336833708,   0.18173822141741],
                [-1.23728482888772,  -0.62770698511448],
                [1.05410903517018,   0.47383850057072],
                [-0.38606826800684,  -0.15947258626548],
                [0.05234403145974,   0.02034464240245]]); # 5th-order coefficients (0<=Gamma<=2.5), col-1 is cylindrical, col-2 is parallel plates - Dec. 3, 2004
            Qrc = np.asarray([
                [1.09524274851589,  1.02903443223445],
                [2.40729067134909,  1.33341005467949],
                [0.06574992723432,  -0.16585375059715],
                [-0.04766884506469,  0.00075548123372]]); #3rd-order coefficients (0<=Gamma<=2.5), col-1 is cylindrical, col-2 is parallel plates - Dec. 3, 2004
            krs0 = 0;
            Qr   = 0;
            A = [1, np.pi**2/16];
            for j in range(krs0c.shape[0]):
                    krs0 = krs0 + krs0c[j,rf]*Gamma**(j);

            for j in range(Qrc.shape[0]):
                    Qr = Qr + Qrc[j,rf]*Gamma**(j);

            kr = krs0/s0;
            E = -A[rf]*Z0*c/np.pi*(np.exp(-kr*s/(2*Qr))*np.cos(kr*s) )/r**2;
        else:
            wp = np.sqrt(Z0*c*sig/tau);
            E  = -Z0*c/np.pi*(np.exp(-s/(4*c*tau))*np.cos(np.sqrt(2*wp/r/c)*s) )/r**2;

    else:
        alf = 0.9;
        a   = 3/(4*np.sqrt(2*np.pi));
        E   = -Z0*c/(4*np.pi)*( (16/3)*np.exp(-s_s0)*np.cos(np.sqrt(3)*s_s0)
                            - (np.sqrt(2*np.pi)*(a**alf+s_s0**(3*alf/2))**(1/alf))**(-1) )/r**2;

    return E

def Eloss(z,I,wake):
    """
    Takes z (meters) a current profile (amps), and a wakefield (?) and generates the ELOSS
    """
    s=z-z[0]
    c  = 2.99792458E8;
    Eloss=np.convolve(wake,I/c,'same')
    return Eloss