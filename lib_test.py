#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:34:02 2019

@author: Nicholas Vieira
"""

import numpy as np 
import matplotlib.pyplot as plt 
import random

import cloud_killer_lib as ck_lib

plt.switch_backend('Qt4Agg')

# In[]:
# TESTING CLOUD_KILLER_LIB

# variables for testing 
TEST_DAY = 697 # the date of interest, if a specific date is desired
NDIM = 8 # the no. of albedo slices
NWALKERS = 100 # the no. of walkers
NSTEPS = 500 # the no. of steps to take
BURNIN = 150 # the no. of steps in the burnin period
PERCENTILE = 84 # 1 or more percentiles (if >1, provide an array)

def run_all(ndim, nwalkers, nsteps, burnin, percentiles=None, datafile=None, 
             day=None):
    """
    Input: The no. of dimensions (albedo slices) in use, the no. of walkers, 
    no. of steps, and burnin step count to use in emcee, the datafile to which
    we wish to write our results (default: no name given), and the EPIC date
    of interest (default: no date given, so a random "great" day is selected.)
    
    Runs all of the various functions in cloud_killer_lib. For debugging. 
    Uncomment the functions you wish to test.
    
    Output: None
    """
    
    if day == None: # if no date specified 
        # load in great days (no nans and more than 17 time points in 24h)
        df = open("great_days.dat", "r")
        contents = df.readlines()
        great_days = [int(c) for c in contents]
        test_date = random.choice(great_days)
        df.close()
    else:
        test_date = day
    print("Day: "+str(test_date))
    print("Date: "+ck_lib.date_after(test_date))

    # Show the raw data
    ck_lib.EPIC_data(test_date)
    
    # Essential: make the MCMC chain
    chaino = ck_lib.make_chain(nwalkers, nsteps, ndim, day=test_date) # MCMC
    
    # MCMC diagnostics 
    ck_lib.plot_walkers_all(chaino) # plot walkers' paths
    ck_lib.cornerplot(chaino, burnin) # corner plot
    
    # MCMC results 
    albmap = ck_lib.mcmc_results(chaino, burnin) # get the map
    print(str(np.round(albmap,3))) # print the params, rounded to 3 decimals
    
    if percentiles != None: 
        # get percentiles
        alb_pers = ck_lib.mcmc_percentiles(chaino, burnin, percentiles) 
        # print percentiles, rounded to 3 decimals
        print(str(np.round(alb_pers,3))) 
    
    # Write to files
    if datafile != None:    
        #ck_lib.mcmc_write(test_date, chaino, burnin, datafile) # mean params 
        if percentiles != None:
            ck_lib.mcmc_write_percentile(test_date, chaino, burnin, 
                                         percentiles, datafile) # percentiles 
    
    # Quality of MCMC fits: plots/Eckert projections
    ck_lib.map_into_fwdmod(chaino, burnin, day=test_date) # one map
    ck_lib.map_into_fwdmod(chaino, burnin, nsamples=50, day=test_date) # multi
    ck_lib.map_into_eckert(albmap, day=test_date) # eckert projection

# random great day:
#run_all(NDIM, NWALKERS, NSTEPS, BURNIN) 
# a specific date, set above 
run_all(NDIM, NWALKERS, NSTEPS, BURNIN, percentiles=PERCENTILE, day=TEST_DAY) 

# In[]:
# WRITING TO FILES 

# Load in "great" days
#df = open("great_days.dat", "r")
#contents = df.readlines()
#great_days = [int(c) for c in contents]
#df.close()

# Writing all of the mean MCMC params to a file
# To use, first comment out mcmc_write_percentile() in run_all()
# And uncomment mcmc_write() 
#for g in great_days:
#    run_all(NDIM, NWALKERS, NSTEPS, BURNIN, datafile="results_all_six.dat", 
#             day=g)

# Writing all of the 16th percentiles to a file
# Comment out mcmc_write() in run_all()
# Uncomment mcmc_write_percentile()
# Set the percentile above 
#for g in great_days:
#    run_all(NDIM, NWALKERS, NSTEPS, BURNIN, PERCENTILE, 
#             "results_all_six_16per.dat", g)
    
# Writing all of the 84th percentiles to a file
# Comment out mcmc_write() in run_all()
# Uncomment mcmc_write_percentile()
#for g in great_days:
#    run_all(NDIM, NWALKERS, NSTEPS, BURNIN, PERCENTILE, 
#             "results_all_six_84per_new.dat", g)