#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:19:28 2019

@author: Nicholas Vieira

To test this script, use the script minima_test.py. 

** Currently only works for 6 OR 8-slice maps. 
"""

import numpy as np
import matplotlib.pyplot as plt
import cloud_killer_lib as ck_lib

class MCMC_results:
    """
    Input: 3 files containing the MCMC results spanning several days. The
    first file should contain the mean MCMC results, the second should contain
    the lower percentile which defines the lower bound on the results, and the 
    third should contain the upper percentile. 
    
    Files should be of the format:
    [day] [param] [param] ... [param]
    [day] [param] [param] ... [param]
    ...
    
    For example: 
    123   0.20156770   0.27899910   ...   0.1980002
    125   0.20190029   0.24311212   ...   0.3032190
    ...
    
    Creates a MCMC_results object which contains all of these results. 
    """
    
    def __init__(self, mean_file, lower_file, upper_file):
        self.mean_file = mean_file      # filename of file with mean results
        self.lower_file = lower_file    # filename with lower percentiles
        self.upper_file = upper_file    # filename with upper percentiles
        self.means = []                 # the actual mean results
        self.lower_pers = []            # actual lower percentiles
        self.upper_pers = []            # actual upper percentiles
        self.days = []                  # days contained in the results 
        
        # load in the mean data            
        df = open(mean_file)
        contents = df.readlines()
        df.close()
        for line in contents:
            array = line.split("\t")
            array.pop() # remove last entry, which is a "\n" character
            array = [float(a) for a in array] # convert to floats
            self.means.append(array) # append all the lines 
            self.days.append(int(array[0])) # append the day of each line
        self.means = np.asarray(self.means) # convert to numpy array
        self.days = np.asarray(self.days) # convert to numpy array
        
        # obtain the dimension of the albedo-space spanned by the results 
        self.ndim = len(array)-1
        
        # load in the lower percentile data            
        df = open(lower_file)
        contents = df.readlines()
        df.close()
        for line in contents:
            array = line.split("\t")
            array.pop() # remove last entry, which is a "\n" character
            array = [float(a) for a in array] # convert to floats
            self.lower_pers.append(array) # append all the lines 
        self.lower_pers = np.asarray(self.lower_pers) # convert to numpy array
        
        # load in the upper percentile data
        df = open(upper_file)
        contents = df.readlines()
        df.close()
        for line in contents:
            array = line.split("\t")
            array.pop() # remove last entry, which is a "\n" character
            array = [float(a) for a in array] # convert to floats
            self.upper_pers.append(array) # append all the lines 
        self.upper_pers = np.asarray(self.upper_pers) # convert to numpy array

class MCMC_batch(MCMC_results):
    """
    Input: the same 3 files as needed by MCMC_results, as well as the 
    days (as integers, e.g. [331,332,334,337]) to create a batch for. 
    
    A sub-class of the MCMC_results class, which spans only a few days. Used
    to obtain minimum-albedo maps over the days provided when instantiating 
    the object.
    """
    def __init__(self, mean_file, lower_file, upper_file, days):
        super(MCMC_batch, self).__init__(mean_file, lower_file, upper_file)
        self.minima = [] # the minimum albedo of each slice over the batch
        self.lower_errs = [] # lower errors on these albedos
        self.upper_errs = [] # upper errors on these albedos
        
        MCMC_batch.obtain_minima(self, days) # automatically obtain everything 
        self.days = days                     # needed to plot minima
        
    def obtain_minima(self, days):
        """
        Input: an array of days for which we want to get the minimum albedo 
        map. Run automatically upon creating an MCMC_batch. 
        Output: None
        """
        
        # verify that the desired days are present 
        valid_days = []
        valid_days_indices = []
        for d in days:
            if d in self.days:
                valid_days.append(d)
                valid_day_ind = int(np.where(self.days == d)[0])
                valid_days_indices.append(valid_day_ind)
            else:
                print("Error: the day %i is not contained in the data. \
                      Ignored."%d)
        print("Valid days among those entered: ",valid_days)
                
        # obtain mean data for days of interest only
        days_of_interest = []
        for i in valid_days_indices:
            days_of_interest.append(self.means[i])
        days_of_interest = np.asarray(days_of_interest) 
        days_of_interest = np.transpose(days_of_interest)[1:] 
        # obtain the minima of the mean of each slice 
        minima = np.min(days_of_interest, axis=1)
        self.minima = minima
        self.means = days_of_interest
        
        # obtain the lower errors on data for days of interest only
        low_pers_of_interest = []
        for i in valid_days_indices:
            low_pers_of_interest.append(self.lower_pers[i])
        low_pers_of_interest = np.asarray(low_pers_of_interest) 
        low_pers_of_interest = np.transpose(low_pers_of_interest)[1:] 
        # obtain the minima of the lower percentile of each slice 
        low_per_minima = np.min(low_pers_of_interest, axis=1)
        self.lower_errs = abs(self.minima - low_per_minima)
        self.lower_pers = low_pers_of_interest
        
        # obtain the upper errors on data for days of interest only
        upper_pers_of_interest = []
        for i in valid_days_indices:
            upper_pers_of_interest.append(self.upper_pers[i])
        upper_pers_of_interest = np.asarray(upper_pers_of_interest) 
        upper_pers_of_interest = np.transpose(upper_pers_of_interest)[1:] 
        # obtain the minima of the upper percentile of each slice 
        upper_per_minima = np.min(upper_pers_of_interest, axis=1)
        self.upper_errs = abs(upper_per_minima - self.minima)
        self.upper_pers = upper_pers_of_interest
        
    
    def plot_minima(self, eckert=False):
        """
        Input: a boolean (default: False) indicating whether to just plot the 
        minimum albedo map or to represent it via an Eckert projection. 
        Output: None
        """
        
        start_day = ck_lib.date_after(self.days[0]) # first day in the batch
        end_day = ck_lib.date_after(self.days[-1]) # last day in the batch 
        
        dayspan = self.days[-1]-self.days[0]+1 # time in days spanned by batch
        days_used = len(self.days) # how many days we actually have data for 
        
        # longitude ticks
        longticks = np.linspace(0,2*np.pi,self.ndim+1)
        longticks = [np.round(l,1) for l in longticks]
        
        
        # It is because of these tick labels that the minima cannot be shown
        # unless the dimension is 6 or 8. If you want to get rid of these tick
        # labels so that you can use other dimensions, comment out these if 
        # statements and the ax.set_... commands below. 
        if self.ndim == 6:
            longtick_labels = ["0", r"$\frac{\pi}{3}$", r"$\frac{2\pi}{3}$", 
                               r"$\pi$", r"$\frac{4\pi}{3}$",
                               r"$\frac{5\pi}{3}$", r"$2\pi$"]
        if self.ndim == 8:
            longtick_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", 
                               r"$\frac{3\pi}{4}$", r"$\pi$", 
                               r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", 
                               r"$\frac{7\pi}{4}$", r"$2\pi$"]
        
        # eckert projection
        if eckert:
            ck_lib.map_into_eckert(self.minima, day=self.days)
            return
        
        # plot of the albedo map with longitude
        # aesthetics
        plt.rcParams.update({'font.size':14})
        fig, ax = plt.subplots()
        fig.set_size_inches((10,6))
        ax.set_xticks(longticks) 
        ax.set_xticklabels(longtick_labels)
        
        # plotting
        phi = np.linspace(0, 2*np.pi*(1-1/self.ndim), len(self.minima))
        # because of differing defns of phi=0, shift arrays before plotting
        minima = np.roll(self.minima, int(self.ndim/2))
        lower_errs = np.roll(self.lower_errs, int(self.ndim/2))
        upper_errs = np.roll(self.upper_errs, int(self.ndim/2))
        errs = [lower_errs, upper_errs]
        plt.plot(phi, minima, marker="o", linestyle="", color="#40a368")
        plt.errorbar(phi, minima, errs, linestyle="", color='black')

        # labels
        plt.xlabel("Longitude "+r"$\phi$")
        plt.ylabel("Albedo "+r"$A$")
        plt.title("Albedo minima from "+start_day+" to "+end_day+
                  " [%d day(s) missing]"%(dayspan-days_used))
        
        # if you like the markers used in my report:
        # for 6-slice data: marker="s", color="#fd5956"
        # for 8-slice data: marker="o", color="#40a368"


        
        
        
        
    
        

            
            
    
    
