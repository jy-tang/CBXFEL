# -*- coding: iso-8859-1 -*-
from __future__ import print_function

import numpy as np
from scipy.optimize import minimize
from scipy.special import *
import pandas as pd
from genesis import genesis
from bisect import bisect_left

# matrices for lattice 
def drift(d):
    return np.asarray([[1,d],[0,1]])

def thicklens(kappa,l):
    phi=np.sqrt(kappa+0*1j)*l
    return np.asarray([[np.cos(phi),1/np.sqrt(kappa+0*1j)*np.sin(phi)],
                       [-np.sqrt(kappa+0*1j)*np.sin(phi),np.cos(phi)]])
def undfocus(Kgen,ku,gamma,l):
    #only for linear undulator!!
    K=Kgen*np.sqrt(2);
    kappa=2*(ku*K/(gamma*2))**2
    return thicklens(kappa,l)

def pd_thicklens(lat,gamma,ku,sign):
    """Convience function which accepts a pandas version of the std lattice"""
    Brho=gamma*0.511/299.8
    kappa=lat['strength']/Brho*sign
    l=lat['L']*(2*np.pi/ku)
    return thicklens(kappa,l)
    
def pd_undfocus(lat,gamma,ku):
    """Convience function which accepts a pandas version of the std lattice"""
    Kgen=lat['strength'] 
    l=lat['L']*(2*np.pi/ku)
    return undfocus(Kgen,ku,gamma,l)
#-------------------------------------------------
# Twiss math
def twissmat(TM):
    #wiedemann 7.62
    C=TM[0][0];S=TM[0][1];
    CP=TM[1][0];SP=TM[1][1];
    r1=[C**2, -2*C*S, S**2]
    r2=[-C*CP, C*SP+CP*S, -S*SP]
    r3=[CP**2, -2*CP*SP, SP**2]
    return np.asarray([r1,r2,r3])

#w,vec = np.linalg.eig(twissmat(np.matmul(drift(1),thicklens(1,0.1))))
def mat_list(mats):
    TM=mats[-1]
    for mat in mats[-2::-1]:
        TM=np.matmul(mat,TM)
    return TM

def matswitch(lat_ele,mat_gamma=None,mat_ku=None):
    typ=lat_ele['type']
    strength=lat_ele['strength']
    L=lat_ele['L']*2*np.pi/mat_ku
    if mat_ku==None: mat_ku=2*np.pi/g.input['xlamd'] #g not defined in function! 
    if mat_gamma==None: mat_gamma=g.input['gamma0']
    matx=drift(L)
    maty=drift(L)
    if typ=='AW':maty=pd_undfocus(lat_ele,mat_gamma,mat_ku)
    if typ=='QF':matx=pd_thicklens(lat_ele,mat_gamma,mat_ku,1);maty=pd_thicklens(lat_ele,mat_gamma,mat_ku,-1)
    if typ=='QD':maty=pd_thicklens(lat_ele,mat_gamma,mat_ku,1);matx=pd_thicklens(lat_ele,mat_gamma,mat_ku,-1)
    return matx,maty

def calc_mat(lat, mat_ku, mat_gamma, fodolength=8):
    # calculate matrix
    TM_X=drift(0)+0*1j
    TM_Y=drift(0)+0*1j
    for ii in range(fodolength):         
        matx,maty=matswitch(lat.iloc[ii],mat_gamma,mat_ku)
        TM_X=np.matmul(matx,TM_X)
        TM_Y=np.matmul(maty,TM_Y)
    return TM_X,TM_Y

def calc_twiss(lat, mat_ku, mat_gamma, fodolength=8):
    TM_X,TM_Y=calc_mat(lat, mat_ku, mat_gamma, fodolength)
    
    # Calculate matched beta functions!
    w,vec = np.linalg.eig(twissmat(TM_X))
    idx=np.argmin(np.abs(np.imag(w)))
    scale=np.sqrt(1/(vec[0,idx]*vec[2,idx]-vec[1,idx]**2))
    keys=['betax','alphax','gammax']
    twiss=dict((key,val) for key,val in zip(keys,np.real(vec[:,idx]*np.abs(scale))))

    w,vec = np.linalg.eig(twissmat(TM_Y))
    idx=np.argmin(np.abs(np.imag(w)))
    scale=np.sqrt(1/(vec[0,idx]*vec[2,idx]-vec[1,idx]**2))
    keys=['betay','alphay','gammay']
    twiss.update([(key,val) for key,val in zip(keys,np.real(vec[:,idx]*np.abs(scale)))])
    
    return twiss

def g_calc_twiss(g,fodolength=8):
    lat=pd.DataFrame.from_dict(split_overlapping(g.lattice))    
    mat_ku=2*np.pi/g.input['xlamd']
    mat_gamma=g.input['gamma0']
    
    twiss=calc_twiss(lat,mat_ku,mat_gamma,fodolength)
    return twiss
#def beta_thick(TM):
        #wiedemann 7.65
        #applies only at symmetery point!
#    C=TM[0][0];S=TM[0][1];
#    CP=TM[1][0];SP=TM[1][1];
#    return S**2/(1-C**2)
#def beta_thin(kappa,l,L,sign):
#    #wiedemann at the center of the quad
#    f=1/(kappa*l)
#    k=f/L/2 #L/2 gives the half-length
#    return L/2*k*(k+sign*1)/np.sqrt(k**2-1)

#-------------------------------------------------
# Functions to change lattice resonant energy (i.e. set AW0, AWD)
def split_overlapping(eles):
    """Splits a (dictionary) standard lattice into distinct elements. Currently fails if quad ends at the same place a drift ends"""
    split_eles=[] #this will be the corrected list of elements
    ele_bounds=[] #list of overlapping element boundaries
    for ii in range(len(eles[1:-1])):    
        ele_bounds.append([eles[ii]['s'],eles[ii]['s']-eles[ii]['L']])
    if len(ele_bounds)>0:                
        slist=[val for sublist in ele_bounds for val in sublist]
        slist=np.sort(np.unique(slist)) #this list now contains the corrected boundaries of elements
        for ss in range(len(slist[1:-1])): #loop through boundaries and assign elements *0th element is a starting point
            for jj in range(len(ele_bounds)):
                if (slist[ss]<=ele_bounds[jj][0])&(slist[ss]>ele_bounds[jj][1]): #if s is in the range of a given element
                    ele=eles[jj].copy()
                    ele['s']=slist[ss]
                    ele['L']=slist[ss]-slist[ss-1] 
                    split_eles.append(ele)
                    break 
    return split_eles

def scale_awd(aw0,L,N=None):
    """
    set awd so that the slippage is a integer N # of wavelengths over drift length L
    if N is none set to min
    """
    if N==None:
        N=np.ceil(L/(1+aw0**2)) #smallest integer for possible scaling
    return np.sqrt((N/L)*(1+aw0**2)-1)

def lat_set_k(lat,K=None,N=None):
    """
    Accepts a dataFrame & modifies AW0 to have strength K & AWD to cause drift N.
    Resonance condition for AWD is guessed from nearest undulator
    """
    if K==None:
        return None
    lat.loc[lat.type=='AW','strength']=K

    for rowindex,ele in lat.loc[lat.type=='AD',:].iterrows():
        if ele['strength']!=0:
            #find closest AW0
            idx=closest(lat.loc[lat.type=='AW','s'].values,ele['s']) 
            aw0=lat.loc[lat.type=='AW','strength'].values[idx]
            L=ele['L']
            lat.loc[rowindex,'strength']=scale_awd(aw0,L,N)
    return None

def g_set_K(g,K=None,N=None):
    """Convience function to set K for a genesis object
    K is the (genesis) undulator strength (AWO). N is the number of periods for AWD to slip (N=None causes minimum slippage)"""
    lat=pd.DataFrame.from_dict(g.lattice)
    if K==None:
        return lat
    lat_set_k(lat,K,N)
    
    g.input['aw0']=np.median( lat.loc[lat.type=='AW','strength'])
    g.input['awd']=np.median( lat.loc[lat.type=='AD','strength'])

    return lat

def g_set_wavelength(g,gamma0=None,energy=None, Nslip=None):
    """Convience function to change the wavelength of the lattice inside the genesis class.
    Energy is in eV. N is the number of periods for AWD to slip (N=None causes minimum slippage)."""
    if gamma0==None: 
        gamma0=g.input['gamma0'];
    else:
        g.input['gamma0']=gamma0
    if energy==None: 
        wavelength=g.input['xlamds'];
    else:
        wavelength=1239.842e-9/energy
        g.input['xlamds']=wavelength
    K0=g.input['aw0'];
   # fun= lambda x:np.abs(g_resonant_wavelength(g,K=x,gamma=gamma0)-wavelength)
   # lbnd=0; ubnd=100;
   # K=sopt.fminbound(fun,lbnd,ubnd,disp=0);
    K=np.sqrt(2*gamma0**2*wavelength/g.input['xlamd']-1)
    print('K: ',K*np.sqrt(2))
    lat=g_set_K(g,K,Nslip)
    g.lattice=pd.DataFrame.to_dict(lat,orient='records')
    # write lattice & reload dict:
    g.write_lattice()
    
#-------------------------------------------------
# scale focusing parameters

def fodo_costfun(split_lat, ku, gamma,fodolength=8):
    """Simple FODO cost function based on the idea that the beta functions are maximized at the entrence to the lattice and half-way through. This will break for more complicated scenerios. Default is a fodo length of 8 account for (U d F d U d D d)"""
    TMX_H,TMY_H=calc_mat(split_lat, ku, gamma, fodolength=fodolength/2)
    twiss=calc_twiss(split_lat, ku, gamma, fodolength=fodolength)
    betax_H=np.abs(np.matmul(twissmat(TMX_H),[[twiss['betax']],[twiss['alphax']],[twiss['gammax']]])[0][0])
    betay_H=np.abs(np.matmul(twissmat(TMY_H),[twiss['betay'],twiss['alphay'],twiss['gammay']])[0])
    costfun=(twiss['betax']**4+twiss['betay']**4+betax_H**4+betay_H**4)**(1/4)
    if np.isnan(costfun): costfun=sys.float_info.max/1e20
    return np.abs(costfun)

def lat_set_Q(lat,f=None,d=None):
    """Change pandas lat to new quad values (f and d should both be positive)"""
    if f!=None:
        lat.loc[((lat.type=='QF')&(lat.strength>0)),'strength']=f
        lat.loc[((lat.type=='QD')&(lat.strength<0)),'strength']=-f
    if d!=None:
        lat.loc[((lat.type=='QD')&(lat.strength>0)),'strength']=d
        lat.loc[((lat.type=='QF')&(lat.strength<0)),'strength']=-d
    return None

def fodo_opt_call(X, lat, ku, gamma):
    """Function to be called by the optimizer"""
    F=X[0]; D=X[1];
    costfun=[]
    lat_set_Q(lat,f=F,d=D)
    cf=fodo_costfun(lat, ku, gamma)
    costfun.append(cf)
    #print(X, costfun)
    return costfun[0]

def fodo_optimization(lat0, ku, gamma, x0,f_bnd=(5,15.3),d_bnd=(5,15.3)):
    """Function to be called by the optimizer: lat0 is the original lattice; ku and gamma are undulator/beam parameters; x0=[f,d] is a guess for the quad strengths, f_ind and d_bnd are bounds."""    
    if x0==None: x0=(lat0.loc[(lat0.type=='QF'),'strength'].max(),-lat0.loc[(lat0.type=='QF'),'strength'].min())
    #print(x0)
    lat=lat0.copy()
    split_lat=pd.DataFrame.from_dict(split_overlapping(pd.DataFrame.to_dict(lat,orient='records'))) 
    res=sopt.minimize(fodo_opt_call,x0=x0,bounds=(f_bnd,d_bnd),args=(split_lat, ku, gamma),                    
                     method='TNC',options={'disp': False,'stepmx':50})
   # print(res)
    print('f={:2.2f}, d={:2.2f}'.format(res.x[0],res.x[1]))
    lat_set_Q(lat0,f=res.x[0],d=res.x[1])
    return None

def g_fodo_optimization(g,lat0=None, ku=None, gamma=None, x0=None,f_bnd=(5,15.3),d_bnd=(5,15.3)):
    """Convienece function to set parameters of fodo_optimization from genesis class g.lat0 is the original lattice; ku and gamma are undulator/beam parameters; x0=[f,d] is a guess for the quad strengths, f_ind and d_bnd are bounds."""
    if lat0==None: 
        lat0=pd.DataFrame.from_dict(g.lattice); 
    if ku==None: 
        ku=2*np.pi/g.input['xlamd'];
    if gamma==None: 
        gamma=g.input['gamma0'];
        
    fodo_optimization(lat0, ku, gamma, x0, f_bnd=f_bnd, d_bnd=d_bnd )
    g.lattice=pd.DataFrame.to_dict(lat0,orient='records')
    g.write_lattice()
    return None
#-------------------------------------------------
def closest(myList, myNumber):
    """
    Returns index of closest element to myNumber in pre-sorted myList.
    Modified from:https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos >= len(myList)-1:
        return len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if np.abs(after - myNumber) < np.abs(myNumber - before):
       return pos
    else:
       return pos-1
