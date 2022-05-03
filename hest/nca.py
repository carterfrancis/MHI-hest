#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numba import jit

@jit('f8[:](f8[:])',nopython=True)
def to_BB(x):
    """
    Converts fractional Brownian noise into a Brownian Bridge
    :param x: input Brownian timeseries as a float numpy 1 D array
    :return: brownian bridge as a float64 numpy 1 D array
    """
    N = x.shape[0]
    k = np.arange(N)
    BB = x - x[0] - (k/(N-1))*(x[N-1] - x[0])
    return BB

@jit('f8[:](f8[:])',nopython=True)
def BB_to_dBB(BB):
    """
    Converts Brownian Bridge into differenced Brownian Bridge (derivative)
    :param BB: input Brownian Bridge a float numpy 1 D array
    :return: differenced brownian bridge as a float64 numpy 1 D array
    """
    dBB = BB[1:] - BB[0:-1]
    return dBB
    
@jit('f8(f8[:],i2,b1)',nopython=True)
def data_AutoCorr(dBB, n, useMean=True):
    """
    Calculates n-th order auto-correlation, given a differenced brownian bridge
    :param dBB: input differneced Brownian Bridge as a float numpy 1 D array
    :param n: which order of autoCorrelation (autocorrelation lag) (int)
    :param useMean: if True, gives unbiased measure of autocorrelation. False will use sum, which gives smaller correlations at high orders
    :return: auto correlation (float64) 
    """
    autoCovN = np.array([dBB[ii]*dBB[ii+n] for ii in range(dBB.shape[0]-n)])
    autoCov0 = np.array([ii**2 for ii in dBB])
    if useMean:
        autoCorr = autoCovN.mean()/autoCov0.mean()
    else:
        autoCorr = autoCovN.sum()/autoCov0.sum()
    return autoCorr

@jit('f8(f8,i2,i2)',nopython=True)
def true_AutoCorr(H, n, N):
    """
    Calulates the theoretical (infinite length timeseries) autocorrelation, given the relevant params
    :param H: hurst exponent of the timeseries (float)
    :param n: which order of autoCorrelation (autocorrelation lag) (int)
    :param N: length of the timeseries (int)
    :return: theoretical auto correlation (float64)
    """
    if n==0:
        return 1.0;
    else:
        if n==1:
            nom = N**(2*H-2) + 1/2*(2**(2*H) -2) + (1- abs(N-1)**(2*H)-abs(N)**(2*H)) / (N*(N-1));
            den = 1-N**(2*H-2);
            return nom / den ;
        else:
            nom = N**(2*H-2) +(1/2)*((n+1)**(2*H)-2*n**(2*H)+(n-1)**(2*H)) +(n**(2*H)-(N -n)**(2*H)-N**(2*H))/(N*(N - n));
            den = 1-N**(2*H-2);
            return nom / den ;
    

def nca(data, seg_l=20, useOddsOnly=False, useMeanCov=True, resolution=0.01, verbose=True):
    """
    Hurst Exponent estimation from timeseries using the Theoretical 1:Nth order autoCorrelation Analysis
    adapted from the following publication:
    Dlask, M., & Kukal, J. (2019). 
    Hurst exponent estimation from short time series. 
    Signal, Image and Video Processing, 13(2), 263-269.
    https://doi.org/10.1007/s11760-018-1353-2
    :param data: mxn matrix of m timeseries each consisting of n timepoints
    :param seg_l: indicates what size of segment to use. Suggested between 5-20
    :param useOddsOnly: whether to use only odd segments. reduces data (increases variability) but avoids correlations between segments
    :param useMeanCov: for unbiased measure of covariance. sum will be used otherwise sum will give smaller correlations at high orders of autocorrelation
    :param resolution: detectable differences in Hurst exponent
    :return: array of m estimated hurst exponents, one for each timeseries
    """
    #convert data into data_mat 3D matrix of m timeseries of n segments of seg_l length
    if useOddsOnly:
        seg_start = np.arange(0,data.shape[1],seg_l*2) #start posns, gives us every other window of the data of length seg_l
    else:
        seg_start = np.arange(0,data.shape[1],seg_l) #start posns, gives us every window of the data of length seg_l
    data_mat = np.zeros((data.shape[0],seg_start.shape[0],seg_l))
    for idx,seg_1 in enumerate(seg_start):
        data_mat[:,idx,:] = data[:,seg_1:seg_1+seg_l]
    
    #turn timeseries (fGn) into fBm
    data_mat = np.cumsum(data_mat, axis=2)  #be careful with axis!!!!!!!!!!!!!!!
    for ii in range(data_mat.shape[0]):
        data_mat[ii,:,:] = (data_mat[ii,:,:].T-data_mat[ii,:,0]).T   #subtract so that time 0 =0
    
    #get dBB (differenced brownian bridge) matrix from timeseries matrix
    if verbose:
        print("getting dBB matrix from timeseries matrix")
    dBB_mat = np.zeros([data_mat.shape[0],data_mat.shape[1],data_mat.shape[2]-1])
    for ii in range(data_mat.shape[0]):
        for jj in range(data_mat.shape[1]):
            dBB_mat[ii,jj,:] = BB_to_dBB(to_BB(data_mat[ii,jj,:]))
    
    #get mxn matrix of m timeseries with n autocorrs (for n timestep jumps)
    if verbose:
        print('get mxn matrix of m timeseries with n autocorrs (for n timestep jumps)')
    autoCorrs = np.zeros([dBB_mat.shape[0],dBB_mat.shape[2]])
    for ii in range(dBB_mat.shape[0]):
        n=0
        autoCorrSum = np.zeros([dBB_mat.shape[2]])
        for jj in range(dBB_mat.shape[1]):
            autoCorrSum += [data_AutoCorr(dBB_mat[ii,jj,:],kk,useMean=useMeanCov) for kk in range(dBB_mat.shape[2])]
            n+=1
        autoCorrs[ii,:] = autoCorrSum / n

    #create array of possible H vals
    h_poss = np.arange(0.01,2,resolution)
    h_poss = h_poss[h_poss!=1]
    #get mxn matrix of true (theoretical) autocorrs where m=hurst n=distance for autocorr
    if verbose:
        print("get mxn matrix of true (theoretical) autocorrs where m=hurst n=distance for autocorr")
    tru_AutoCorrs = np.zeros([h_poss.shape[0], autoCorrs.shape[1]])
    for ii in range(h_poss.shape[0]):
        tru_AutoCorrs[ii,:] = [true_AutoCorr(h_poss[ii], jj, dBB_mat.shape[2]) for jj in range(autoCorrs.shape[1])]
    
    #optimize over H's
    if verbose:
        print("Optimize over H's")
    h_final = np.zeros(data_mat.shape[0])
    for ii in range(autoCorrs.shape[0]):
        diff = autoCorrs[ii,:]-tru_AutoCorrs
        diffSquared = diff*diff
        matchCost = diffSquared.sum(1) #value of cost function we would like to minimize
        h_final[ii] = h_poss[np.argmin(matchCost)]
        
    return h_final
        
