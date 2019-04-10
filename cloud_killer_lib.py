#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:13:38 2019

@author: nvieira
"""

# modules which comes with all (most) python installs
import numpy as np 
import scipy.optimize as op
import matplotlib.pyplot as plt 
import random
import math

# modules which do not 
from netCDF4 import Dataset
import emcee
import corner
import cartopy.crs as ccrs
from astropy.time import Time

plt.switch_backend('Qt4Agg')

# RETRIEVE DATA
data = Dataset("dscovr_single_light_timeseries.nc")
data.dimensions.keys()
radiance = data.variables["normalized"][:] # lightcurves for different wavelengths

# Constants used throughout
SOLAR_IRRAD_780 = 1.190 # Units: W m^-2 nm^-1
FILTER_WIDTH_780 = 1.8 # According to DSCOVR paper; units: nm
#FILTER_WIDTH_780 = 2.0 # According to DSCOVR-MODIS calibration paper; units: nm

# Constant arrays used throughout
RAD_780 = radiance[9] # lightcurves for 780 nm
#time in seconds starting June 13, 2015 00:00:00 UTC
TIME_SECS = radiance[10]
#time in days since June 13, 2015  00:00:00 UTC
TIME_DAYS = TIME_SECS/86148.0 #86148 = 23.93h

#longitude at SOP/SSP: convert UTC at SOP/SSP to longitude 
#longitude is 0 (2pi) at t=0 and decreases with time
SOP_LONGITUDE = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0))%(2*np.pi) for t in TIME_SECS]
#longitude in degrees rather than radians
SOP_LONGITUDE_DEG = [l*180.0/np.pi for l in SOP_LONGITUDE]

# In[ ]:
def EPIC_data(day, plot=True):
    """
    Input: a date (int) after 13 June 2015 00:00:00, a boolean indicating 
    whether or not to plot the data
    Output: time, longitude (deg), apparent albedo, error on apparent albedo, 
    bool indicating if dataset contains NaNs
    """
    # starting on the desired day
    n=0
    while (TIME_DAYS[n] < day):
        n += 1 # this n is where we want to start
        
    # EPIC takes data between 13.1 to 21.2 times per day
    # need to import ~22 observations and then truncate to only one day
    t = TIME_DAYS[n:n+22]
    longitude = SOP_LONGITUDE_DEG[n:n+22]
    flux_rad = RAD_780[n:n+22] # Units: W m^-2 nm^-1
    reflectance = flux_rad*np.pi/SOLAR_IRRAD_780 # Units: unitless

    # truncate arrays to span one day only
    while ((t[-1] - t[0]) > 1.0):   # while t spans more than one day
        t = t[0:-2]                 # truncate arrays 
        longitude = longitude[0:-2]
        flux_rad = flux_rad[0:-2]
        reflectance = reflectance[0:-2]
 
    # error on reflectance
    reflectance_err = 0.02*reflectance # assuming 2% error     
    # add gaussian noise to the data with a variance of up to 2% mean reflectance
    gaussian_noise = np.random.normal(0, 0.02*np.mean(reflectance), len(reflectance))
    reflectance += gaussian_noise
    
    contains_nan = False # check for nans in data
    number_of_nans = 0
    for f in flux_rad:
        if math.isnan(f) == True:
            number_of_nans += 1
            contains_nan = True     
    if contains_nan:
        print("CAUTION: "+str(number_of_nans)+" points in this set are NaN")
        return t, longitude, reflectance, reflectance_err, contains_nan
     
    if plot:
        # plotting reflectance over time
        fig = plt.figure()
        ax1 = fig.add_subplot(111)    
        ax1.errorbar(t, reflectance, yerr=reflectance_err, fmt='.', 
                     markerfacecolor="cornflowerblue", 
                     markeredgecolor="cornflowerblue", color="black")
        ax1.set_ylabel("Apparent Albedo "+r"$A^*$", size=18)
        ax1.set_xlabel("T-minus 13 June 2015 00:00:00 UTC [Days]", size=18)
        
        plt.title("EPIC data"+r" ["+r"$d = $"+date_after(day)
                                           +", $\phi_0 = $"+"%.1f]"%longitude[0])
        plt.rcParams.update({'font.size':14})
        plt.show()

    return t, longitude, reflectance, reflectance_err, contains_nan
    # this "reflectance" is probably actually the apparent albedo

# In[]:
# FORWARD MODEL
    
def kernel(longitude, phi_obs):
    """
    Input: an array of longitudes and the SOP phi
    Output: the kernel
    """
    # I=V in this case since the SOP and SSP are the same at L1, but we choose 
    # to fix sub-observer/sub-stellar latitude at pi/2 
    V = np.cos(longitude[...,np.newaxis] - phi_obs)
    V[V<0] = 0 #set values <0 to be = 0

    return V*V # K=I*V=V*V


def lightcurve(albedos, time_days=1.0, long_frac=1.0, n=10000, phi_obs_0 = 0.0, 
               plot=False, alb=False): 
    """
    Input: an array of albedos, the time in days which the model should span
    (default: 1.0), the longitude (as a fraction of 2pi) which the model should 
    span (default: 1.0), the no. of points n to generate (default 10000), 
    the initial sub-observer longitude (default: 0.0), a 
    boolean indicating whether or not to plot the lightcurve (default: False), 
    and a boolean indicating whether to return the reflectance or to 
    apply the multiplicative factor of 3/2 such that the lightcurve's units 
    match those of EPIC data
    
    Output: the lightcurve, in units of reflectance or apparent albedo 
    """
    C = 4.0/(3.0*np.pi) # integration constant
    
    # n times between 0 and 23.93 h, in h
    time = np.linspace(0.0, time_days*23.93, n, False)
    # len(albedos) longitudes
    phi = np.linspace(2*np.pi, 0, len(albedos), False) 
    
    w_Earth = 2.0*np.pi/23.93 # Earth's angular velocity 
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    # phi decreases before returning to 0 in this convention
    
    albedos = np.asarray(albedos) # convert to numpy array
    
    kern = kernel(phi, phi_obs) # compute the kernel  
    
    reflectance = np.sum(albedos[...,np.newaxis]*kern, axis=0)
    reflectance = C*reflectance*(2*np.pi)/len(albedos)
    
    if alb: # if we want units in agreement with EPIC data
        reflectance *= 3.0/2.0 # multiply by 3/2
        
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)    
        ax1.plot(time, reflectance, color='red')
        if alb:
            ax1.set_ylabel("Apparent Albedo "+r"$A^*$")
        else:
            ax1.set_ylabel("Reflectance")
        ax1.set_xlabel("Time [h]")
        plt.show()
    
    return time, reflectance


def fit_EPIC(day, plot_raw=False, verbose=False):
    """
    Input: a day of interest in the EPIC data, a boolean indicating whether to
    plot the raw data separately, and a boolean indicating whether to print 
    the timespan, phi-span, and initial phi of the fit.
    
    Fits the chosen day of EPIC data with the forward model (plug'n'chug.)
    
    Output: None
    """
    # obtain EPIC data
    t, phi, ref, ref_err, nans = EPIC_data(day, plot_raw)
    
    timespan = t[-1] - t[0] # time (in days) spanned by EPIC data 
    phispan = timespan # phi (as a frac of 2pi) spanned by EPIC data
    
    if verbose:
        print("FITTING EPIC DATA:")
        print("Timespan: %f days"%timespan)
        print("Phi-span: %f*2pi"%phispan)
        print("Initial "+r"$\phi_o$ = %f deg"%phi[0])
    
    # fit the model to the EPIC data
    phi_obs_init = phi[0]*np.pi/180.0 # the initial phi_obs in rad
    # generate the light curve with 10,000 points, but do not plot
    model_time, model_ref = lightcurve(ref, timespan, phispan, 10000, 
                                       phi_obs_init, plot=False, alb=True)
    
    fig, ax = plt.subplots()
    ax.errorbar(np.linspace(0,23.93*timespan,len(t)), ref, yerr=ref_err, 
                fmt='.', label="EPIC data", markerfacecolor="cornflowerblue", 
                markeredgecolor="cornflowerblue", color="black")
    ax.plot(model_time, model_ref, label="Forward model", color='red')
    ax.set_ylabel("Apparent Albedo"+r" $A^*$")
    ax.set_xlabel("Time [h]")
    ax.set_title("EPIC fit - plugging into forward model"+
                 r" ["+r"$d = $"+date_after(day)+
                 ", $\phi_0 = $"+"%.1f]"%phi[0])
    plt.rcParams.update({'font.size':18})
    plt.legend()
    
def fit_EPIC_maxlike(day, ndim, plot_raw=False, verbose=False):
    """
    Input: a day of interest in the EPIC data, the number of slices in the
    albedo map to be produced, a boolean indicating whether to
    plot the raw data separately, and a boolean indicating whether to print 
    the timespan, phi-span, and initial phi of the fit.
    
    Fits the chosen day of EPIC data by maximizing the likelihood.
    
    Output: None
    """
    # obtain EPIC data
    t, phi, ref, ref_err, nans = EPIC_data(day, plot_raw)
    
    timespan = t[-1] - t[0] # time (in days) spanned by EPIC data 
    phispan = timespan # phi (as a frac of 2pi) spanned by EPIC data
    
    if verbose:
        print("FITTING EPIC DATA:")
        print("Timespan: %f days"%timespan)
        print("Phi-span: %f*2pi"%phispan)
        print("Initial "+r"$\phi_o$ = %f deg"%phi[0])
    
    # fit the model to the EPIC data
    phi_obs_init = phi[0]*np.pi/180.0 # the initial phi_obs in rad
    
    # get the albedo parameters which maximize the likelihood
    alb_guess = [0.25 for i in range(ndim)]
    fit_params = opt_lnlike(alb_guess, t, ref, ref_err) # maximize the likelihood 
    print(fit_params)
    
    # generate the light curve with 10,000 points, but do not plot
    model_time, model_ref = lightcurve(fit_params, timespan, phispan, 10000, 
                                       phi_obs_init, plot=False, alb=True)
    
    fig, ax = plt.subplots()
    ax.errorbar(np.linspace(0,23.93*timespan,len(t)), ref, yerr=ref_err, 
                fmt='.', label="EPIC data", markerfacecolor="cornflowerblue", 
                markeredgecolor="cornflowerblue", color="black")
    ax.plot(model_time, model_ref, label="Maximum likelihood forward model", 
            color='#632de9')
    ax.set_ylabel("Apparent Albedo"+r" $A^*$")
    ax.set_xlabel("Time [h]")
    ax.set_title("EPIC fit - maximized likelihood"+
                 r" ["+r"$d = $"+date_after(day)+
                 ", $\phi_0 = $"+"%.1f]"%phi[0])
    plt.rcParams.update({'font.size':18})
    plt.legend()


# In[]:
# STATISTICS 

def chisq_calc(data, data_err, model, reduced=False, verbose=False):
    """ 
    Input: data, error on data, a model to data, a boolean speciying which 
    chisq to return and a boolean indicating whether to print the reduced chisq
    Output: chisq OR rchisq
    
    * CURRENTLY UNUSED. *
    
    """
    chisq_num = np.power(np.subtract(data,model), 2) # (data-model)**2
    chisq_denom = np.power(data_err, 2) # (error)**2
    chisq = sum(chisq_num/chisq_denom)
    rchisq = chisq/len(data) # reduced chisq aka chisq per datum
    
    if verbose and reduced:
        print("mean chisq per datum = ",np.mean(rchisq))
    if reduced:
        return rchisq
    return chisq

    
def lnlike(alpha, time, ref, ref_err):
    """
    Input: an array of albedos to feed into the forward model, array of times 
    of measurements and arrays of reflectances and errors on reflectances.
    
    Feeds the albedos into the forward model and produces a model; compares 
    the measurements to the data, then assesses the likelihood of a set of 
    given observations. Likelihood assessed using chisq. 
    
    Output: ln(likelihood)
    """
    
    # time/longitude spanned by forward model
    timepts = len(time) # no. of time points
    timespan = (time[-1]-time[0]) # time spanned, in days
    phispan = timespan # longitude spanned, as a fraction of 2pi
    
    # obtain model prediction, in units of apparent albedo
    model_time, model_ref = lightcurve(alpha, timespan, phispan, timepts, 
                                       0, plot=False, alb=True) 
    
    # compute ln(likelihood)
    chisq_num = np.power(np.subtract(ref,model_ref), 2) # (data-model)**2
    chisq_denom = np.power(ref_err, 2) # (error)**2
    res = -0.5*sum(chisq_num/chisq_denom + np.log(2*np.pi) + np.log(np.power(
            ref_err,2))) #lnlike
    
    return res


def opt_lnlike(alpha, time, ref, ref_err):
    """
    Input: Guesses for the parameters to be fit for (alpha, an array of 
    albedos) and the time, light curve, and error on the light curve of the
    data being fit 
    
    Maximizes the ln(likelihood).
    
    Output: the values of albedos with maximum likelihood
    """
    nll = lambda *args: -lnlike(*args) # return -lnlike of args
    # boundaries on the possible albedos:
    bound_alb = tuple((0.000001,0.999999) for i in range(len(alpha))) 
    # minimize (-ln(like)) to maximimize the likelihood 
    result = op.minimize(nll, alpha, args=(time,ref,ref_err), bounds=bound_alb)
    
    return result['x'] # the optimized parameters


def lnprior(alpha):
    """
    Input: Array of guesses for albedos of the slices
    Output: The ln(prior) for a given set of albedos 
    """
    if np.all(alpha>0.0) and np.all(alpha<1.0):
        return 0.0
    return -np.inf


def lnpost(alpha, time, ref, ref_err):
    """
    Input: guesses for albedos (an array alpha) and data to which we compare 
    (time, reflectance, and error on reflectances)
    Output: ln(posterior)
    """
    lp = lnprior(alpha)
    if not np.isfinite(lp): # if ln(prior) is negative inf (prior->0) 
        return -np.inf      # then ln(post) is negative inf too
    return lp + lnlike(alpha, time, ref, ref_err)

# In[]:
# EMCEE 
    
def init_walkers(alpha, time, ref, ref_err, ndim, nwalkers):
    """
    Input: the guesses for albedo alpha, the data (time, ref, ref_err) needed
    to optimize the likelihood, the number of dimensions (i.e., albedo 
    slices to be fit), and the number of walkers to initialize
    
    Initializes the walkers in an albedo-space in a gaussian ball centered 
    on the parameters which maximize the likelihood 
    
    Output: the initial positions of all walkers in the ndim-dimensional 
    parameter space
    """
    opt_albs = opt_lnlike(alpha, time, ref, ref_err)
    pos = [opt_albs + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    return pos


def make_chain(day, nwalkers, nsteps, ndim):
    """
    Input: the day in the EPIC data of interest, the number of albedo slices 
    (parameters) being fit, the number of walkers, and the number of steps 
    to take in the chain
    
    Runs MCMC on EPIC data for the given day of interest. 
    
    Output: an emcee sampler object's chain
    """
    t, phi, r, r_err, nans = EPIC_data(day, False) # get data
    
    # guess: alb is 0.25 everywhere
    init_guess = np.asarray([0.25 for n in range(ndim)])
    # better guess: maximize the likelihood
    opt_params  = opt_lnlike(init_guess, t, r, r_err) 
    
    # initialize nwalkers in a gaussian ball centered on the opt_params
    init_pos = init_walkers(opt_params, t, r, r_err, ndim, nwalkers)
    
    # set up the sampler object and run MCMC 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err))
    sampler.run_mcmc(init_pos, nsteps)
    return sampler.chain

def make_chain_artificial(alpha, nwalkers, nsteps, ndim):
    """
    Input: a synthetic array of albedo data, the number of albedo slices 
    (parameters) being fit, the number of walkers, and the number of steps 
    to take in the chain
    Output: an emcee sampler object's chain
    """
    t, r = lightcurve(alpha, 1.0, 1.0, 10000, alb=True) # get synthetic data
    
    r_err = 0.02*r # assuming 2% error     
    # add gaussian noise to the data with a variance of up to 2% mean app alb
    gaussian_noise = np.random.normal(0, 0.02*np.mean(r), len(r))
    r += gaussian_noise
    
    # guess: alb is 0.25 everywhere
    init_guess = np.asarray([0.25 for n in range(ndim)])
    # better guess: maximize the likelihood
    opt_params  = opt_lnlike(init_guess, t, r, r_err) 
    
    # initialize nwalkers in a gaussian ball centered on the opt_params
    init_pos = init_walkers(opt_params, t, r, r_err, ndim, nwalkers)
    
    # set up the sampler object and run MCMC 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err))
    sampler.run_mcmc(init_pos, nsteps)
    return sampler.chain

def flatten_chain(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    Output: a flattened chain, ignoring all steps pre-burnin
    """
    ndim = len(chain[0][0]) # number of params being fit 
    return chain[:,burnin:,:].reshape(-1, ndim)

def walker_paths_1dim(chain, dimension):
    """
    Input: an emcee sampler chain and the dimension (parameter) of interest.
    
    Builds an array of arrays, where each entry in the array represents a 
    a single walker and each subarray contains the path taken by a particular 
    walker in parameter space. 
    
    Output: (nwalker x nsteps) 2D array of paths for each walker
    """
    
    ndim = len(chain[0][0])
    # if user asks for a dimension larger than the number of params we fit
    if (dimension >  (ndim-1)): 
        print("\nWarning: the input chain is only %d-dimensional. Please \
              input a number between 0 and %d. Exiting now."%(ndim,(ndim-1)))
        return
        
    nwalkers = len(chain)  # number of walkers
    nsteps = len(chain[0]) # number of steps taken

    # obtain the paths of all walkers for some dimension (parameter)
    walker_paths = []
    for n in range(nwalkers):
        single_path = [ chain[n][s][dimension] for s in range(nsteps) ]
        walker_paths.append(single_path)
    return walker_paths

def plot_walkers_1dim(chain, dimension):
    """
    Input: a chain produced by emcee and the dimension (parameter, beginning 
    at 0 and ending at ndim-1) we wish to plot
    
    Plots the paths of all walkers for a single dimension (parameter) in the 
    given chain. 
    
    Output: None
    """

    ndim = len(chain[0][0])
    # if user asks for a dimension larger than the number of params we fit
    if (dimension >  (ndim-1)): 
        print("\nWarning: the input chain is only %d-dimensional. Please \
              input a number between 0 and %d. Exiting now."%(ndim,(ndim-1)))
        return
        
    nwalkers = len(chain)  # number of walkers
    nsteps = len(chain[0]) # number of steps taken

    step_number = [x for x in range(1, nsteps+1)] # steps taken as an array
    
    for n in range(nwalkers):
        y = [ chain[n][s][dimension] for s in range(nsteps) ]
        plt.plot(step_number, y)
        
        
def plot_walkers_all(chain):
    """
    Input: an emcee sampler chain
    
    Plots the paths of all walkers for all dimensions (parameters). Each 
    parameter is represented in its own subplot.
    
    Output: None
    """
    nsteps = len(chain[0]) # number of steps taken
    ndim = len(chain[0][0]) # number of params being fit
    step_number = [x for x in range(1, nsteps+1)] # steps taken as an array
    
    # plot the walkers' paths
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.1)
    for n in range(ndim):   # for each param
        paths = walker_paths_1dim(chain, n) # obtain paths for each param
        fig.add_subplot(ndim,1,n+1) # add a subplot for the param
        plt.xlabel("Steps")
        #plt.twiny()
        for p in paths:
            plt.plot(step_number, p,color='k',alpha=0.3) # plot all walker paths
            plt.ylabel(r"$A$"+"[%d]"%(n)) # label parameter

def cornerplot(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Produces a corner plot for the fit parameters. 
    
    Output: None
    """
    ndim = len(chain[0][0]) # number of params being fit
    samples = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    label_albs = [] # setting the labels for the corner plot
    for n in range(ndim):
        label_albs.append(r"$A$"+"[%d]"%(n))
    
    plt.rcParams.update({'font.size':12})
        
    corner.corner(samples, labels=label_albs, quantiles=(0.16, 0.5, 0.84), 
                  levels=(1-np.exp(-0.5),))
    
# In[]:
# MCMC RESULTS 
    
def mcmc_results(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Averages the parameters of each dimension in parameter space to obtain 
    the results 
    Output: an array of the mean parameters (albedos) found via MCMC
    """
    ndims = len(chain[0][0]) # obtain no. of dimensions
    flat = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    mcmc_params = []
    for n in range(ndims): # for each dimension
        param_n_temp = []
        for w in range(len(flat)):
            param_n_temp.append(flat[w][n])
        mcmc_params.append(np.mean(param_n_temp))
    return mcmc_params

def mcmc_percentiles(chain, burnin, pers=[16,50,84]):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    Output: an array of the percentiles desired by the user (default: 16th, 
    50th, and 84th) found by MCMC, with dimension = 3 x (no. of 
    albedo sices)
    """
    ndims = len(chain[0][0]) # obtain no. of dimensions
    flat = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    mcmc_percentiles = []
    for n in range(ndims): # for each dimension 
        percentile_n_temp = []
        for w in range(len(flat)): 
            percentile_n_temp.append(flat[w][n]) 
        mcmc_percentiles.append(np.percentile(percentile_n_temp, pers, axis=0))
    return mcmc_percentiles
    
def mcmc_write(day, chain, burnin, output_file=None):
    """
    Input: the day being fit, an emcee sampler chain, the steps taken during 
    the burnin, and the name of an output file (optional; set automatically
    if none is given)
    
    Writes the parameters returned by MCMC to a tab-delimited file for 
    later loading in. Each column represents a slice. This function is designed 
    to be run many times to then compare several days of data. 
    
    Output: None
    """
    mcmc_params = mcmc_results(chain, burnin)
    
    if output_file == None: # if user does not give a specific output filename
        output_file = "mcmc_results_"+str(day) 
       
    df = open(output_file,"a+") # append to an existing file or create it
    line = str(day)+"\t" # first, the day being fit
    for i in range(len(mcmc_params)):
        line += str(mcmc_params[i])+"\t" # next, all of the albedo parameters
    line += "\n"
    df.write(line)
    df.close()
    
def mcmc_write_percentile(day, chain, burnin, pers=50, output_file=None):
    """
    Input: the day being fit, an emcee sampler chain, the steps taken during 
    the burnin, and the name of an output file (optional; set automatically
    if none is given)
    
    Writes the given percentile (default 50th, i.e., the median) returned by 
    MCMC to a tab-delimited file for later loading in. Each column represents
    a slice. This function is designed to be run many times to then compare
    several days of data.
    
    Output: None
    """
    
    mcmc_pers = mcmc_percentiles(chain, burnin, pers)
    
    if output_file == None: # if user does not give a specific output filename
        output_file = "mcmc_results_"+str(day)+"_"+str(pers)
       
    df = open(output_file,"a+") # append to an existing file or create it
    line = str(day)+"\t" # first, the day being fit
    for i in range(len(mcmc_pers)):
        line += str(mcmc_pers[i])+"\t" # next, all of the computed percentiles
    line += "\n"
    df.write(line)
    df.close()
    

# In[]:
# VISUALIZING THE RESULTS
    
def map_into_fwdmod(chain, burnin, nsamples=None, day=None):
    """
    Input: an emcee sampler chain, the steps taken during the burnin, the 
    number of random samples to take (default None), and the day of EPIC data 
    which we are fitting (default is None, in which case, we are plotting 
    the MCMC results for artificial data)
    
    If nsamples is None (default), plots only the mean MCMC results. If an 
    integer is given, takes nsamples random samples and plots them too as 
    almost-transparent black lines  
    
    Output: None
    """
    
    mean_mcmc_params = mcmc_results(chain, burnin) # mean values of MCMC results
    
    # if we are fitting one day of EPIC data
    if day != None: 
        t, phi, ref, ref_err, nans = EPIC_data(day, False)
        timespan = t[-1] - t[0] # time spanned by EPIC data (days)
        phispan = timespan  # phi (as frac of 2pi) spanned by EPIC data
        
        phi_obs_init = phi[0]*np.pi/180.0
        phi_obs_init = 0
        mean_mcmc_time, mean_mcmc_ref = lightcurve(mean_mcmc_params, timespan, 
                                                       phispan, 10000, 
                                                       phi_obs_init, plot=False,
                                                       alb=True)
        
        # a series of random samples    
        flat = flatten_chain(chain, burnin) # get a flattened chain to sample
        sample_params = flat[np.random.randint(len(flat),size=nsamples)] # samples
        
        fig, ax = plt.subplots()
        ax.errorbar(np.linspace(0,23.93*timespan,len(t)), ref, yerr=ref_err, fmt='.', 
                    label="EPIC data", markerfacecolor="cornflowerblue", 
                    markeredgecolor="cornflowerblue", color="black")
        
        if nsamples != None:
            # a series of random samples    
            flat = flatten_chain(chain, burnin) # get a flattened chain to sample
            sample_params = flat[np.random.randint(len(flat),size=nsamples)] # samples
            for s in sample_params: 
                sample_time, sample_ref = lightcurve(s, timespan, phispan, 
                                                     10000, phi_obs_init, 
                                                     plot=False, alb=True)
                
                ax.plot(sample_time, sample_ref, color='k', alpha=0.1, 
                        label="Random samples (%d)"%nsamples)
        
        ax.plot(mean_mcmc_time, mean_mcmc_ref, label="Mean MCMC parameters", 
                    color='red')
        ax.set_ylabel("Apparent Albedo "+r"$A^*$")
        ax.set_xlabel("Time [h]")
        ax.set_title("EPIC fit - MCMC"+r" ["+r"$d = $"+date_after(day)
                                           +", $\phi_0 = $"+"%.1f]"%phi[0])
    
        if nsamples != None:
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[-1], handles[-2], handles[0]]
            labels = [labels[-1], labels[-2], labels[0]]
            plt.legend(handles, labels)
        else:
            plt.legend()
        
    # if we are fitting an artificial lightcurve     
    else: 
        mean_mcmc_time, mean_mcmc_ref = lightcurve(mean_mcmc_params, 1.0, 
                                                       1.0, 10000, 
                                                       0, plot=False,
                                                       alb=True)
        # a series of random samples    
        flat = flatten_chain(chain, burnin) # get a flattened chain to sample
        sample_params = flat[np.random.randint(len(flat),size=nsamples)] # samples
        
        fig, ax = plt.subplots()
        
        if nsamples != None:
            # a series of random samples    
            flat = flatten_chain(chain, burnin) # get a flattened chain to sample
            sample_params = flat[np.random.randint(len(flat),size=nsamples)] # samples
            for s in sample_params:
                sample_time, sample_ref = lightcurve(s, 1.0, 1.0, 
                                                         10000, 0, plot=False,
                                                         alb=True)
                ax.plot(sample_time, sample_ref, color='k', alpha=0.1, 
                        label="Random samples (%d)"%nsamples)
                
        ax.plot(mean_mcmc_time, mean_mcmc_ref, label="Mean MCMC parameters", 
                    color='red')
        ax.set_ylabel("Apparent Albedo "+r"$A^*$")
        ax.set_xlabel("Time [h]")
        ax.set_title("Artificial data - MCMC"+r" ["+"$\phi_0 = 0$]")
        
        if nsamples != None: 
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[-1], handles[-2]]
            labels = [labels[-1], labels[-2]]
            plt.legend(handles, labels)
        else: 
            plt.legend()
    
def map_into_eckert(params, nlats=200, nlons=200, day=None):
    """
    Input: the params from MCMC, the no. of latitude points to use (200 by 
    default for resolution purposes), and the number of longitude points to 
    use (200 by default so that the gradient between different filled 
    contours is sharp), and the date of interest for the EPIC data, if desired
    
    Produces a map of the Earth with filled countours corresponding to the 
    albedo of a given slice. 
    
    ** CURRENTLY ONLY WORKS FOR 6 OR 8 SLICES
    
    Output: None
    """
    
    # if we are showing EPIC results from ONE date
    if type(day) in [int, float]:
        t, phi, ref, ref_err, nans = EPIC_data(day, False) 
        # cartopy longitude decreases as the planet spins and is 0 at Greenwich
        # the forward model does the same but is 0 at 180 deg West from Greenwich
        # we compute the offset from the definition of 0 in the forward model
        phi_offset = np.deg2rad(360-phi[0]) 
        
        lons = np.linspace(2*np.pi-phi_offset, -phi_offset, nlons) # longitude spanned

    # if we are showing multi-day real data or artificial results    
    else: 
        lons = np.linspace(2*np.pi, 0, nlons)
    
    nslices = int(len(params)) # longitudinal slices in the map
    interval = int(nlons/nslices)  # entries in longitude array which span one slice

    lats = np.linspace(-np.pi/2, np.pi/2, nlats)

    lats, lons = np.meshgrid(lats, lons)
    lats = np.rad2deg(lats)
    lons = np.rad2deg(lons)
    
    # eckert IV projection
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1, projection=ccrs.EckertIV())
    
    data = []
    
    # if 6 slices:
    if nslices == 6:
        for i in range(nlons):
            if (i <= interval):
                temp = [params[0] for i in range(nlats)]
            elif (i <= 2*interval):
                temp = [params[1] for i in range(nlats)]
            elif (i <= 3*interval):
                temp = [params[2] for i in range(nlats)]
            elif (i <= 4*interval):
                temp = [params[3] for i in range(nlats)]
            elif (i <= 5*interval):
                temp = [params[4] for i in range(nlats)]
            elif (i <= 6*interval):
                temp = [params[5] for i in range(nlats)]
            data.append(temp)
    
    # if 8 slices:
    if nslices == 8:
        for i in range(nlons):
            if (i <= interval):
                temp = [params[0] for i in range(nlats)]
            elif (i <= 2*interval):
                temp = [params[1] for i in range(nlats)]
            elif (i <= 3*interval):
                temp = [params[2] for i in range(nlats)]
            elif (i <= 4*interval):
                temp = [params[3] for i in range(nlats)]
            elif (i <= 5*interval):
                temp = [params[4] for i in range(nlats)]
            elif (i <= 6*interval):
                temp = [params[5] for i in range(nlats)]
            elif (i <= 7*interval):
                temp = [params[6] for i in range(nlats)]
            elif (i <= 8*interval):
                temp = [params[7] for i in range(nlats)]
            data.append(temp)
       
    # make the map using a Plate Carree transformation
    cs = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(),
                     cmap='gist_gray', alpha=0.3)
    
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Albedo $A$') # make the color bar appear
    ax.coastlines() # draw coastlines
    ax.set_global() # make the map global
    
    if type(day) in [int, float]: # if one day
        plt.title("EPIC albedo map - MCMC"+r" ["+r"$d = $"+date_after(day)+"]")
        
    elif type(day) == list: # if several days, as when obtaining minima
        start_day = date_after(day[0]) # first date
        end_day = date_after(day[-1]) # final date
        dayspan = day[-1]-day[0] # time in days spanned by map
        days_used = len(day) # how many days we actually have data for 
        plt.title("Albedo minima from "+start_day+" to "+end_day+
                  " [%d day(s) missing]"%(dayspan-days_used))
        plt.rcParams.update({'font.size':14})
        
    else: # if just artificial data 
        plt.title("Albedo map - Eckert IV Projection")

# In[]:
# UTILITY
        
def date_after(d):
    """
    Input: an integer d
    
    Useful to quickly find out the actual calendar date of some day 
    in the EPIC dataset. 
    
    Output: the date, d days after 2015-06-13 00:00:00.000
    """
    
    t_i = Time("2015-06-13", format='iso', scale='utc')  # make a time object
    t_new_MJD = t_i.mjd + d # compute the Modified Julian Day (MJD) of the new date
    t_new = Time(t_new_MJD, format='mjd') # make a new time object
    t_new_iso = t_new.iso # extract the ISO (YY:MM:DD HH:MM:SS) of the new date
    t_new_iso = t_new_iso.replace(" 00:00:00.000", "") # truncate after DD
    
    return t_new_iso 
    
# In[]:
# TESTING
### Testing MCMC ###

TEST_DAY = 415 # the date of interest, if a specific date is desired
NDIM = 6 # the no. of albedo slices
NWALKERS = 200 # the no. of walkers
NSTEPS = 300 # the no. of steps to take
BURNIN = 50 # the no. of steps in the burnin period
PERCENTILES = [16,84] # 1 or more percentiles (if >1, provide an array)

def test_all(ndim, nwalkers, nsteps, burnin, percentiles=None, datafile=None, 
             day=None):
    """
    Input: The no. of dimensions (albedo slices) in use, the no. of walkers, 
    no. of steps, and burnin step count to use in emcee, the datafile to which
    we wish to write our results (default: no name given), and the EPIC date
    of interest (default: no date given, so a random "great" day is selected.)
    
    Tests all of the various functions above. For debugging. Uncomment the 
    functions you wish to test.
    
    Output: None
    """
    
    if day == None: # if no date specified 
        # load in great days (no nans and more than 17 time points in 24h)
        df = open("good_days.dat", "r")
        contents = df.readlines()
        great_days = [int(c) for c in contents]
        test_date = random.choice(great_days)
        df.close()
    else:
        test_date = day
    print("Day: "+str(test_date))
    print("Date: "+date_after(test_date))

    # Show the raw data
    #EPIC_data(test_date)
    
    # Show raw data & fit it by just plugging in to the forward model
    #fit_EPIC(test_date)
    
    # Essential: make the MCMC chain
    chaino = make_chain(test_date, nwalkers, nsteps, ndim) # MCMC
    
    # MCMC diagnostics 
    #plot_walkers_all(chaino) # plot walkers' paths
    #cornerplot(chaino, burnin) # corner plot
    
    # MCMC results 
    #albmap = mcmc_results(chaino, burnin) # get the map
    #print(str(np.round(albmap,3))) # print the params, rounded to 3 decimals
    
    #if percentiles != None: 
    #    alb_pers = mcmc_percentiles(chaino, burnin, percentiles) # get percentiles
    #    print(str(np.round(alb_pers,3))) # print percentiles, rounded to 3 decimals
    
    # Write to files
    if datafile != None:    
        mcmc_write(test_date, chaino, burnin, datafile) # mean params 
    #    if percentiles != None:
    #        mcmc_write_percentile(test_date, chaino, burnin, percentiles, datafile) # percentiles 
    
    # Quality of MCMC fits: plots/Eckert projections
    #map_into_fwdmod(chaino, burnin, day=test_date) # one map
    #map_into_fwdmod(chaino, burnin, nsamples=50, day=test_date) # plot multiple maps
    #map_into_eckert(albmap, day=test_date) # plot an eckert projection

#test_all(NDIM, NWALKERS, NSTEPS, BURNIN) # random great day
#test_all(NDIM, NWALKERS, NSTEPS, BURNIN, percentiles=PERCENTILES, day=TEST_DAY) # a specific date, set above 
 

# In[]:
# WRITING TO FILES 

# Load in "great" days
df = open("great_days.dat", "r")
contents = df.readlines()
great_days = [int(c) for c in contents]
df.close()

# Writing all of the mean MCMC params to a file
# To use, first comment out mcmc_write_percentile()
# And uncomment mcmc_write() 
#for g in great_days:
#    if g >= 325:
#        test_all(NDIM, NWALKERS, NSTEPS, BURNIN, datafile="results_all_six.dat", 
#             day=g)

# Writing all of the 16th percentiles to a file
# Comment out mcmc_write()
# Uncomment mcmc_write_percentile()
# Set the percentile above 
#for g in great_days:
#    test_all(NDIM, NWALKERS, NSTEPS, BURNIN, "results_all_16per.dat", g)
    
# Writing all of the 84th percentiles to a file
# Comment out mcmc_write()
# Uncomment mcmc_write_percentile()
# Set the percentile above 
#for g in great_days:
#    test_all(NDIM, NWALKERS, NSTEPS, BURNIN, "results_all_84per.dat", g)

# In[]:
# OTHER TESTS #

### Testing quality of a fit obtained by maximizing our likelihood ###
#alb_guess = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
#t, phi, ref, ref_err, nanners = EPIC_data(415,False) 
#fit_params = opt_lnlike(alb_guess, t, ref, ref_err) # maximize the likelihood 
#phi_init = phi[0]*np.pi/180.0
#fit_time, fit_alb = lightcurve(fit_params, 1, 1, 10000, phi_init, alb=True)
#plt.plot(fit_time, fit_alb, marker='.', linestyle='')
#plt.plot((t-t[0])*23.93, ref, marker='.', linestyle='')

#fit_EPIC_maxlike(415, 8)
    








     















