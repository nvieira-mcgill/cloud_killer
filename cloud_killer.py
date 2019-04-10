#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:19:28 2019

@author: nvieira
"""

import numpy as np
import matplotlib.pyplot as plt
import cloud_killer_lib as ck_lib

results = "results_all.dat"
results_16per = "results_all_16per.dat"
results_84per = "results_all_84per.dat"

# batches to use
batches = []
batches.append([90,92,94,96,97,98,99])
batches.append([112,113,115,118,119,120,121,122,124,128])
batches.append([321,323,324,325,327,328,329,330,331])
batches.append([335,337,339])
batches.append([345,349,350,351,352,353,354,355,356,357,358,359])
batches.append([364,365,366,367,369,370,372,373,374,375,376,379,380,382,383])


class MCMC_results:
    def __init__(self, mean_file, lower_file, upper_file):
        self.mean_file = mean_file
        self.lower_file = lower_file
        self.upper_file = upper_file
        self.means = []
        self.lower_pers = []
        self.upper_pers = []
        self.days = []
        
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
    
    def __init__(self, mean_file, lower_file, upper_file, days):
        super(MCMC_batch, self).__init__(mean_file, lower_file, upper_file)
        self.minima = []
        self.lower_errs = []
        self.upper_errs = []
        
        MCMC_batch.obtain_minima(self, days)    # automatically obtain everything 
        self.days = days                        # needed to plot minima
        
    def obtain_minima(self, days):
        # verify that the desired days are present 
        valid_days = []
        valid_days_indices = []
        for d in days:
            if d in self.days:
                valid_days.append(d)
                valid_day_ind = int(np.where(self.days == d)[0])
                valid_days_indices.append(valid_day_ind)
            else:
                print("Error: the day %i is not contained in the data. Ignored."%d)
        print("Valid days among those entered: ",valid_days)
                
        # obtain mean data only for days of interest
        days_of_interest = []
        for i in valid_days_indices:
            days_of_interest.append(self.means[i])
        days_of_interest = np.asarray(days_of_interest) # convert to 
        days_of_interest = np.transpose(days_of_interest)[1:] 
        # obtain the minima of the mean of each slice 
        minima = np.min(days_of_interest, axis=1)
        self.minima = minima
        self.means = days_of_interest
        
        # obtain the lower errors on data only for days of interest
        low_pers_of_interest = []
        for i in valid_days_indices:
            low_pers_of_interest.append(self.lower_pers[i])
        low_pers_of_interest = np.asarray(low_pers_of_interest) # convert to 
        low_pers_of_interest = np.transpose(low_pers_of_interest)[1:] 
        # obtain the minima of the lower percentile of each slice 
        low_per_minima = np.min(low_pers_of_interest, axis=1)
        self.lower_errs = self.minima - low_per_minima
        self.lower_pers = low_pers_of_interest
        
        # obtain the upper errors on data only for days of interest
        upper_pers_of_interest = []
        for i in valid_days_indices:
            upper_pers_of_interest.append(self.upper_pers[i])
        upper_pers_of_interest = np.asarray(upper_pers_of_interest) # convert to 
        upper_pers_of_interest = np.transpose(upper_pers_of_interest)[1:] 
        # obtain the minima of the lower percentile of each slice 
        upper_per_minima = np.min(upper_pers_of_interest, axis=1)
        self.upper_errs = upper_per_minima - self.minima
        self.upper_pers = upper_pers_of_interest
        
    
    def plot_minima(self, eckert=False):
        
        start_day = ck_lib.date_after(self.days[0])
        end_day = ck_lib.date_after(self.days[-1])
        
        dayspan = self.days[-1] - self.days[0] # time in days spanned by batch
        days_used = len(self.days) # how many days we actuall have data for 
        
        # longitude ticks
        longticks = np.linspace(0,2*np.pi,self.ndim+1)
        longticks = [np.round(l,1) for l in longticks]
        
        # eckert projection
        if eckert:
            ck_lib.map_into_eckert(self.minima, day=self.days)
            return
        
        # plot of the albedo map with longitude  
        plt.rcParams.update({'font.size':14})
        fig, ax = plt.subplots()
        fig.set_size_inches((10,5))
        ax.set_xticks(longticks)
        phi = np.linspace(0, 2*np.pi, len(self.minima))
        errs = [self.lower_errs, self.upper_errs]
        plt.plot(phi, self.minima, marker="o", linestyle="", color="#40a368")
        plt.errorbar(phi, self.minima, errs, linestyle="", color='black')

        plt.xlabel("Longitude "+r"$\phi$")
        plt.ylabel("Albedo "+r"$A$")
        plt.title("Albedo minima from "+start_day+" to "+end_day+
                  " [%d day(s) missing]"%(dayspan-days_used))

counter = 0
for b in batches:
    counter += 1
    res = MCMC_batch(results, results_16per, results_84per, b)
    res.plot_minima(True)
    #plt.savefig("albedo_min_"+str(counter)+".png")
    plt.savefig("albedo_min_eck"+str(counter)+".png")
counter = 0
        
        
        
        
        
    
        

            
            
    
    