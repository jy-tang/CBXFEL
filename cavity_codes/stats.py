# -*- coding: iso-8859-1 -*-


import numpy as np
import matplotlib.pyplot as plt


def fwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None, return_more=False):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.max(inds[(inds < arg_max) * (scaled < 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.min(inds[(inds > arg_max) * (scaled < 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        import matplotlib.pyplot as plt
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    if return_more:
        return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])
    else:
        return np.abs(xhigh-xlow)


# maximum width at half max
def mwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None, return_more=False):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.min(inds[(inds < arg_max) * (scaled > 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.max(inds[(inds > arg_max) * (scaled > 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    if return_more:
        return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])
    else:
        return np.abs(xhigh-xlow)


# maximum width at half max
def rms(array):
    x = np.arange(len(array))
    xm = np.dot(x,array) / np.sum(array)
    x2m = np.dot(x**2,array) / np.sum(array)
    return np.sqrt(x2m-xm**2)


# maximum width at half max
def mean(array):
    x = np.arange(len(array))
    xm = np.dot(x,array) / np.sum(array)
    return xm
    
    