# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import SDDS
import sys, os, csv, copy

"""
run sdds2genesis.py
filename = 'LCLS2scS.out'
sdds2genesis(filename, nbins=300, core_region_mask_type=3, savePlotQ=False, showPlotQ=True, mask_threshold_relative=0.75)
"""

def silogbase(rawnumber):
    log10rn = np.log10(rawnumber)
    signlog10rn = np.sign(log10rn)
    return signlog10rn*np.int(np.floor(np.abs(log10rn))/3+0.5*(1-signlog10rn))*3
def sibase(rawnumber):
    return 10**silogbase(rawnumber)
def sipref(rawnumber):
    si_prefixes = {-21:['zepto','z'],-18:['atto','a'],-15:['femto','f'],-12:['pico','p'],-9:['nano','n'],-6:['micro','u'],-3:['milli','m'],0:['',''],3:['kilo','k'],6:['mega','M'],9:['giga','G'],12:['tera','T'],15:['peta','P'],18:['exa','E']}
    return si_prefixes[silogbase(rawnumber)][0]
def sip(rawnumber):
    si_prefixes = {-21:['zepto','z'],-18:['atto','a'],-15:['femto','f'],-12:['pico','p'],-9:['nano','n'],-6:['micro','u'],-3:['milli','m'],0:['',''],3:['kilo','k'],6:['mega','M'],9:['giga','G'],12:['tera','T'],15:['peta','P'],18:['exa','E']}
    return si_prefixes[silogbase(rawnumber)][1]

# testsi = 2e15; print(testsi, sibase(testsi), silogbase(testsi), sipref(testsi), sip(testsi))
# testsi = 2; print(testsi, sibase(testsi), silogbase(testsi), sipref(testsi), sip(testsi))
# testsi = 2e-15; print(testsi, sibase(testsi), silogbase(testsi), sipref(testsi), sip(testsi))

# !pwd
# !ls

def find_core_region_1d(array, threshold): 
    # takes a 1D array and finds coords bounding the region about the peak
    cut = array > threshold
    imax = array.argmax()
    ihi = imax; ilo = imax
    for i in range(imax-1, -1, -1):
        if cut[i]:
            ilo = i
        else:
            break
    for i in range(imax, len(array)):
        if cut[i]:
            ihi = i
        else:
            break
    return [ilo, ihi]

def mask_core_region(array, threshold_relative=0.001):
    # takes a 1D or 2D array and masks the image with the core region along one axis
    mask = 0*array
    if len(np.shape(array)) is 1: # 1D array
        r = find_core_region_1d(array,np.max(array)*threshold_relative)
        mask[r[0]:r[1]+1] = 1
    else:
        for i in range(len(array)):
            mask[i] = mask_core_region(array[i], threshold_relative)
    return mask

def mask_core_region2(array, threshold_relative=0.001):
    # takes a 2D array and masks the image with the core region along both axes
    return mask_core_region(array, threshold_relative)*mask_core_region(array.T, threshold_relative).T

def find_core_region2r(array, threshold_relative=0.001):
    # takes a 2D array and masks the image with the core region along both axes
    mask = mask_core_region(array, threshold_relative)*mask_core_region(array.T, threshold_relative).T
    masked_array = array * mask
    sum0 = np.sum(masked_array, axis=0)
    bounds0 = find_core_region_1d(sum0, np.max(sum0)*threshold_relative)
    sum0 = np.sum(masked_array, axis=1)
    bounds1 = find_core_region_1d(sum0, np.max(sum0)*threshold_relative)
    return np.array([bounds0, bounds1])

def mask_core_region2r(array, threshold_relative=0.001):
    # takes a 2D array and masks the image with the core region along both axes
    mask = 0*array
    cr = find_core_region2r(array, threshold_relative)
#     mask[cr[0,0]:cr[0,1]+1,cr[1,0]:cr[1,1]+1] = 1
    mask[cr[1,0]:cr[1,1]+1,cr[0,0]:cr[0,1]+1] = 1
    return mask
    
def plot_tps(dist, filenamehead, nbins=300, savePlotQ=True, showPlotQ=False):
    
    cwd = os.path.realpath(os.path.curdir)
    
    xwind = dist[0].max() - dist[0].min()
    ywind = dist[2].max() - dist[2].min()
    
    h, xedges, yedges, image = plt.hist2d(dist[0,:]/sibase(xwind),dist[1,:],bins=nbins);
    plt.xlabel('x position ('+sip(xwind)+'m)'); plt.ylabel('x\''); plt.colorbar(); 
    if savePlotQ: plt.savefig(os.path.join(cwd,filenamehead+'_xvsxp.png'), bbox_inches='tight')
    if showPlotQ: plt.show()
    plt.close()
    
    h, xedges, yedges, image = plt.hist2d(dist[2,:]/sibase(ywind),dist[3,:],bins=nbins);
    plt.xlabel('y position ('+sip(ywind)+'m)'); plt.ylabel('y\''); plt.colorbar(); 
    if savePlotQ: plt.savefig(os.path.join(cwd,filenamehead+'_yvsyp.png'), bbox_inches='tight')
    if showPlotQ: plt.show()
    plt.close()
    
def match_to_FODO(dist, dist_core, L_quad=10*0.026, L_drift=150*0.026, g_quad=14.584615):
    
    d_all = dist # 
    d_core = dist_core # core of the distribution to match
    
    # calculate mean energy, emittance, and twiss
    
    gamma0 = np.mean(d_core[5,:])
    
    xxp_cov = np.cov(d_core[:2,:])
    #xxp_cov = np.cov(d[:2,:])
    xemit = np.sqrt(np.linalg.det(xxp_cov))
    xtwiss = xxp_cov / xemit
    xrms_core = np.sqrt(xxp_cov[0,0])
    xprms_core = np.sqrt(xxp_cov[1,1])
    
    yyp_cov = np.cov(d_core[2:4,:])
    yemit = np.sqrt(np.linalg.det(yyp_cov))
    ytwiss = yyp_cov / yemit
    yrms_core = np.sqrt(yyp_cov[0,0])
    yprms_core = np.sqrt(yyp_cov[1,1])
    
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
    
    average_beamsize_core = np.sqrt(xrms_match * yrms_match)
    emittance_normalized_core = np.sqrt(xemit * yemit) * gamma0
    
    #print('betaMAX = ', betaMAX)
    #print('betaMIN = ', betaMIN)
    #print('xrms_match = ', xrms_match)
    #print('yrms_match = ', yrms_match)
    #print('xprms_match = ', xprms_match)
    #print('yprms_match = ', yprms_match)
    #print('xemit = ', xemit)
    #print('yemit = ', yemit)

    # match the beam
    
    # copy the beam to transform and return
    #d_match = 1.*d_core;
    d_match = 1.*d_all;
    
    # correct first moments
    for i in range(4):
        d_match[i,:] -= np.mean(d_core[i,:])
        
    # calculate matching matricies
    # M: sigma_match == M.sigma_core.(M^T)
    Mmx = np.zeros([2,2])
    Mmx[0,0] = xrms_match * xprms_core / xemit
    Mmx[0,1] = - xxp_cov[0,1] * xrms_match / xprms_core / xemit
    Mmx[1,1] = xprms_match / xprms_core
    Mmy = np.zeros([2,2])
    Mmy[0,0] = yrms_match * yprms_core / yemit
    Mmy[0,1] = - yyp_cov[0,1] * yrms_match / yprms_core / yemit
    Mmy[1,1] = yprms_match / yprms_core
    
    # correct second moments
    #print('sqrt diag cov = ', np.sqrt(np.diag(np.cov(d_match[:2,:]))))
    #print('xprms_core, xrms_core = ', xprms_core, xrms_core)
    #print('xcov = ', np.cov(d_match[:2,0,:]))
    d_match[0,:] = Mmx[0,0] * d_match[0,:] + Mmx[0,1] * d_match[1,:]
    d_match[1,:] = Mmx[1,1] * d_match[1,:]
    #print('xcov = ', np.cov(d_match[:2,:]))
    #print('ycov = ', np.cov(d_match[2:4,:]))
    d_match[2,:] = Mmy[0,0] * d_match[2,:] + Mmy[0,1] * d_match[3,:]
    d_match[3,:] = Mmy[1,1] * d_match[3,:]
    #print('ycov = ', np.cov(d_match[2:4,:]))
    
    return d_match, average_beamsize_core, emittance_normalized_core

# BROKEN for core_region_mask_type = 0 (need to put some try/except statements regarding h0)
# consider passing an array of axes to plot to and then make a function which generates
# a grid of axes and plots a series of particle phase spaces
def sdds2genesis(filename, nbins=300, core_region_mask_type=3, savePlotQ=True, showPlotQ=False, mask_threshold_relative=0.75, genesis_ndmax=int(1.25e6), R56_fs=0): 
    # NOTE: ONLY USE core_region_mask_type == 3
    # core_region_mask_type: 0 => none; 1 => 1D masking; 2 => 2D masking; 3 => bound to core in 2D
    # options 2 and 3 need work (is there an axis 0,1 confusion on the cutting? maybe check with non-square matricies)
    # ndmax is 1250000 in stock Genesis, but can be increased if modified in source code and recompiled
    genesis_ndmax = int(genesis_ndmax)
    
    mc2_eV = 0.511e6
    fnhead = '.'.join(filename.split('.')[:-1])
    #fnhead = fnhead.split('/')[-1]
    #fnhead = fnhead.split('\\')[-1]
    #fnhead = './' + fnhead
    cwd = os.path.realpath(os.path.curdir)
    
    # load data
    ff = SDDS.readSDDS(filename)
    parameters, bunches = ff.read()
    print('number of bunches: ', bunches.shape[0],' \t particles in bunch 0: ', bunches.shape[1])
    d = bunches[0].T # particle data
    npart = bunches.shape[1] # number of macroparticles
    charge_pC = parameters[0]['Charge'] * 1e12

    # apply r56
    if R56_fs is not 0:
         d[4,:] -= R56_fs * 1e-15 * (d[5,:] / d[5].mean() - 1.)
    
    # center in time
    d[4,:] -= np.mean(d[4,:])
    
    # longitudinal phase space
    
    tmean = d[4].mean()
    twind = d[4].max() - d[4].min()
    Emean = d[5].mean() * mc2_eV

    # cut = np.arange(npart); np.random.shuffle(cut); cut = cut < 1000
    # plt.scatter(1e12*(d[4,cut]-tmean),0.511*d[5,cut],s=1);
    # plt.xlabel('s (ps)'); plt.ylabel('Energy (MeV)'); plt.show(); plt.close()

    # cut particles for managable plots
    cut = np.arange(npart); np.random.shuffle(cut); cut = cut < 10000000
    d_ts = (d[4,cut]-tmean)/sibase(twind)
    d_gammas = mc2_eV*d[5,cut]/sibase(Emean)
    h, xedges, yedges, image = plt.hist2d(d_ts,d_gammas,bins=nbins); 
    plt.xlabel('Time ('+sip(twind)+'s)'); plt.ylabel('Energy ('+sip(Emean)+'eV)'); plt.colorbar(); 
    if savePlotQ: plt.savefig(os.path.join(cwd,fnhead+'_lps.png'), bbox_inches='tight')
    if showPlotQ: plt.show()
    plt.close()
    

    # core the beam
    xedges0 = xedges; yedges0 = yedges; charge_core_frac = 1.
    if core_region_mask_type == 1:
        h0 = h * mask_core_region(h, threshold_relative=mask_threshold_relative)
    if core_region_mask_type == 2:
        h0 = h * mask_core_region2(h, threshold_relative=mask_threshold_relative)
    if core_region_mask_type > 0:
        if core_region_mask_type == 3:
            cr = find_core_region2r(h.T, threshold_relative=mask_threshold_relative)
            h0 = h[cr[0,0]:cr[0,1]+1,cr[1,0]:cr[1,1]+1]
            xedges0 = xedges[cr[0,0]-1:cr[0,1]+1]
            yedges0 = yedges[cr[1,0]-1:cr[1,1]+1]
        extent0 = [min(xedges0), max(xedges0), min(yedges0), max(yedges0)]
        plt.imshow(np.fliplr(h0).T, extent=extent0);
        plt.gca().set_aspect((extent0[1]-extent0[0])/(extent0[-1]-extent0[-2])/1.25)
        plt.xlabel('Time ('+sip(twind)+'s)'); plt.ylabel('Energy ('+sip(Emean)+'eV)'); plt.colorbar(); 
        if savePlotQ: plt.savefig(os.path.join(cwd,fnhead+'_lps_core.png'), bbox_inches='tight')
        if showPlotQ: plt.show()
        plt.close()
        charge_core_frac = np.sum(h0)/np.sum(h)

    # plot transverse phase space

    # TODO: interpolate selected region as a function of time with default to cut all particles
    if core_region_mask_type == 3:
        
        if False:
            plot_tps(d, fnhead, nbins=nbins, savePlotQ=savePlotQ, showPlotQ=showPlotQ)
        
        print('min(d_ts) = ',min(d_ts),'; mean(d_ts) = ',np.mean(d_ts),'; max(d_ts) = ', max(d_ts),'; std(d_ts) =',np.std(d_ts))
        print('extent0 =', extent0)
        
        # core on the bounded area
        coresel = (d_ts > extent0[0]) * (d_ts < extent0[1]) * (d_gammas > extent0[2]) * (d_gammas < extent0[3])
        
        # or don't 
        coresel = d_ts * 0. == 0. #################### NOT CORE - SELECTING ALL #################
        
        ## select just the horn
        #hornts = d[4,:] * 1e15; hornts -= np.mean(hornts)
        #hornxs = d[0,:] * 1e6;
        #hornxps = d[1,:] * 1e6;
        #coresel = np.abs(hornts-14.5) < 2.
        #coresel *= np.abs(hornxs - 375.) < 50.
        
        d_core = d[:,coresel]
        gamma_core = np.mean(d_core[5,:])
        
        plot_tps(d_core, fnhead+'_core', nbins=nbins, savePlotQ=savePlotQ, showPlotQ=showPlotQ)
        
#         return d_core
    
    # match beam
    #d_match, average_beamsize_core, emittance_normalized_core = match_to_FODO(d, d_core, L_quad=0.3, L_drift=3.6, g_quad=12.64)
    
    # match beam but then rematch just orbit
    d_match, average_beamsize_core, emittance_normalized_core = match_to_FODO(d, d,L_quad=10*0.026, L_drift=150*0.026, g_quad=14.584615)
    #d_match[0,:] -= np.mean(d_match[0,coresel]) # match horn position
    #d_match[1,:] -= np.mean(d_match[1,coresel]) # match horn angle
    
    # select just the horn
    plot_tps(d_match[:,coresel], fnhead+'_matched_core', nbins=nbins, savePlotQ=savePlotQ, showPlotQ=showPlotQ)
    
    
    if True:
        plot_tps(d_match, fnhead+'_matched', nbins=nbins, savePlotQ=savePlotQ, showPlotQ=showPlotQ)
    
    #return d_core, d_match
    
    # cut low current tails
    
    # current profile 
    xcoords = 0.5*(xedges[1:]+xedges[:-1])
    ycoords = np.sum(h,axis=1) / np.sum(h) * charge_pC / np.mean(np.diff(xcoords))
    xcoords0 = 0.5*(xedges0[1:]+xedges0[:-1])
    ycoords0 = np.sum(h0,axis=1) / np.sum(h0) * charge_pC / np.mean(np.diff(xcoords0)) * charge_core_frac
    plt.plot(xcoords, ycoords, label='All');
    plt.plot(xcoords0, ycoords0, label='Core'); plt.legend()
    plt.xlabel('Time ('+sip(twind)+'s)'); plt.ylabel('Current ('+sip(ycoords.max())+'A)'); 
    if savePlotQ: plt.savefig(os.path.join(cwd,fnhead+'_current.png'), bbox_inches='tight')
    if showPlotQ: plt.show()
    plt.close()
    #import copy
    #espread = copy.copy(ycoords)
    #espread_core = copy.copy(ycoords0)
    charge_cumulant = np.cumsum(ycoords); charge_cumulant /= charge_cumulant[-1]
    #keep_charge_fraction = 0.7; head_keep_padding_fs = 20.; tail_keep_padding_fs = 0.
    #keep_current_fraction= 0.5; head_keep_padding_fs = 10.; tail_keep_padding_fs = 0.
    keep_current_fraction= 0.5; head_keep_padding_fs = 15.; tail_keep_padding_fs = 10.
    #keep_charge_fraction = 0.85; head_keep_padding_fs = 20.; tail_keep_padding_fs = 0.
    #keep_charge_fraction = 0.85; head_keep_padding_fs = 40.; tail_keep_padding_fs = -10.
    #keep_charge_fraction = 0.9; head_keep_padding_fs = 10.; tail_keep_padding_fs = 0.
    #charge_cut = np.abs(2. * charge_cumulant - 1.) < keep_charge_fraction
    #kept_ts = xcoords[charge_cut] * sibase(twind)
    dxcoords = xcoords[1]-xcoords[0]
    from scipy.signal import medfilt
    filterwidth_fs = 20.
    filterwidth = np.int(filterwidth_fs/dxcoords); filterwidth = 1+2*np.int(filterwidth/2)
    #print('filterwidth_fs = ', filterwidth_fs)
    #print('dxcoords = ', dxcoords)
    #print('filterwidth = ', filterwidth)
    ycoordsmf = medfilt(ycoords, kernel_size=filterwidth)
    #kept_ts = xcoords[ycoords/np.max(ycoords) > keep_current_fraction] * sibase(twind)
    kept_ts = xcoords[ycoordsmf/np.max(ycoordsmf) > keep_current_fraction] * sibase(twind)
    #print('sibase(twind)=',sibase(twind))
    #print('min(kept_ts) = ', min(kept_ts))
    #print('max(kept_ts) = ', max(kept_ts))
    keep_t_min = min(kept_ts) - head_keep_padding_fs * 1e-15
    keep_t_max = max(kept_ts) + tail_keep_padding_fs * 1e-15
    #print('keep_t_min = ', keep_t_min)
    #print('keep_t_max = ', keep_t_max)
    #print('np.mean(d_match[4,:]) = ', np.mean(d_match[4,:]))
    #print('np.median(d_match[4,:]) = ', np.median(d_match[4,:]))
    #print('np.std(d_match[4,:]) = ', np.std(d_match[4,:]))
    keep_t_selection = (d_match[4,:] > keep_t_min) * (d_match[4,:] < keep_t_max)
    d_match = d_match[:,keep_t_selection] # cut the distribution
    
    # ANOTHER IDEA: look at 2D density histo, medfilt in 2D, newprof=max(array,axis=0); cut on newprof
    # ANOTHER IDEA: look at 2D density histo, cut below density threshold, newprof=sum(array,axis=0); cut on newprof
    
    plt.hist2d(d_match[4,:]/sibase(twind), mc2_eV*d_match[5,:]/sibase(Emean), 300); 
    plt.xlabel('Time ('+sip(twind)+'s)'); plt.ylabel('Energy ('+sip(Emean)+'eV)'); #plt.colorbar(); 
    ax=plt.gca().twinx()
    #ax.plot(xcoords * sibase(twind), ycoords/np.max(ycoords), label='All');
    #ax.plot(xcoords * sibase(twind), xcoords*0+keep_current_fraction, label='All');
    #ax.plot(xcoords * sibase(twind), ycoordsmf/np.max(ycoordsmf), label='All');
    ax.plot(xcoords, ycoords/np.max(ycoords), label='All');
    ax.plot(xcoords, xcoords*0+keep_current_fraction, label='All');
    ax.plot(xcoords, ycoordsmf/np.max(ycoordsmf), label='All');
    ax.set_ylabel('Rel. currents and cuts')
    if savePlotQ: plt.savefig(os.path.join(cwd,fnhead+'_lps_distfile.png'), bbox_inches='tight')
    if showPlotQ: plt.show()
    plt.close()
    
    # current profile 
    
    # dump distribution to distfile for genesis
    
    # select particles randomly but in order
    npart_match = np.shape(d_match)[1]
    cut_ndmax = np.arange(npart_match); np.random.shuffle(cut_ndmax); cut_ndmax = cut_ndmax < genesis_ndmax
    indicies_ndmax = np.arange(npart_match)[cut_ndmax]
    beam_sel_charge = charge_pC * npart_match / npart *1e-12
    #print(keep_charge_fraction, '=?', 1. * npart_match / npart)
    
    header = ["# sdds2genesis from elegant matched for undulator line "]
    header += ["? version = 1.0"]
    header += ["? charge =   " + str(beam_sel_charge)]
    header += ["? size =   " + str(genesis_ndmax)]
    header += ["? COLUMNS X XPRIME Y YPRIME T GAMMA"]
    
    distfilename = fnhead+'_matchproj.dist'
    with open(distfilename, 'w') as distfile:
        for line in header:
            distfile.write(line+'\n')
            
        writer = csv.writer(distfile, delimiter=" ") # genesis requires space delimiters but can also do "\t "
        for i in indicies_ndmax:
            writer.writerow(d_match[:6,i])
    
    print('INFO: Wrote ',distfilename)
    
    beam_description = {} # probably core params
    beam_description['gamma'] = gamma_core
    beam_description['x_rms'] = average_beamsize_core
    beam_description['emittance_normalized'] = emittance_normalized_core
    beam_description['time_range'] = keep_t_max - keep_t_min
    #beam_description['current'] = 
    #beam_description['gamma_rms'] = 
    
    return beam_description

def main():
    
    showPlotQ = False
    if(len(sys.argv) > 1):
        if sys.argv[1] == '-p' or sys.argv[1] == '-sp' or sys.argv[1] == '--plot' or sys.argv[1] == '--showplot':
            showPlotQ = True
            filenames = sys.argv[2:]
        else:
            filenames = sys.argv[1:]
    else:
        print('Please specify a filename.')
        return 1
    
    nbins = 300
    
    for filename in filenames:
        #sdds2genesis(filename, nbins=nbins, core_region_mask_type=1, savePlotQ=True, showPlotQ=showPlotQ, R56_fs=0, mask_threshold_relative=0.25)
        sdds2genesis(filename, nbins=nbins, core_region_mask_type=3, savePlotQ=True, showPlotQ=showPlotQ, R56_fs=0)
#         try:
#             print('Processing ',filename)
#             sdds2genesis(filename, nbins=nbins, core_region_mask_type=3, savePlotQ=True, showPlotQ=showPlotQ, R56_fs=0)
#         except:
#             print('Something went wrong processing file', filename)
            
    return 0

if __name__== "__main__":
    main()
    
