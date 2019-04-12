#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:19:28 2019

@author: nvieira
"""

import numpy as np
import matplotlib.pyplot as plt
import cloud_killer_lib as ck_lib

# 8-slice results
results = "results_all.dat"
results_16per = "results_all_16per.dat"
results_84per = "results_all_84per.dat"

# 6-slice results
#results = "results_all_six.dat"
#results_16per = "results_all_six_16per.dat"
#results_84per = "results_all_six_84per.dat"

# batches to use
# this was my choice, but you can pick any set of days you want 
batches = []
batches.append([90,92,94,96,97,98,99])
batches.append([112,113,115,118,119,120,121,122,124,128])
batches.append([321,323,324,325,327,328,329,330,331])
batches.append([335,337,339]) # this batch throws an error for 6 slices 
batches.append([345,349,350,351,352,353,354,355,356,357,358,359])
batches.append([364,365,366,367,369,370,372,373,374,375,376])
batches.append([379,380,382,383])
batches.append([390,391,393,394,395,396,397,398,399,400,401])
batches.append([402,404,405,408,409,410,411])
batches.append([414,415,416,417,418,421,422,423,424,425,426,427])
batches.append([428,429,431,432,433])
batches.append([436,437,438,439,440,441,442,443,444,446])
batches.append([684,685,686,689,690,691])
batches.append([695,696,697,699,700,701,702,703,704,705])
batches.append([706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,
                721,722,723,724])
batches.append([730,731,732,733,734,735,736,737,738,739])
batches.append([740,741,742,743,744,745,746,747,748,749])
batches.append([750,751,752,754,755,757,758])
batches.append([761,762,763,764,765,766,767,768])
batches.append([770,772,773,774,775,777,778,779])
batches.append([780,781,782,784,785,786,788,789])
batches.append([790,791,792,793,794,795,797,798,799])
batches.append([805,807,808,809,811,812,813])



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
        days_of_interest = np.asarray(days_of_interest) 
        days_of_interest = np.transpose(days_of_interest)[1:] 
        # obtain the minima of the mean of each slice 
        minima = np.min(days_of_interest, axis=1)
        self.minima = minima
        self.means = days_of_interest
        
        # obtain the lower errors on data only for days of interest
        low_pers_of_interest = []
        for i in valid_days_indices:
            low_pers_of_interest.append(self.lower_pers[i])
        low_pers_of_interest = np.asarray(low_pers_of_interest) 
        low_pers_of_interest = np.transpose(low_pers_of_interest)[1:] 
        # obtain the minima of the lower percentile of each slice 
        low_per_minima = np.min(low_pers_of_interest, axis=1)
        self.lower_errs = abs(self.minima - low_per_minima)
        self.lower_pers = low_pers_of_interest
        
        # obtain the upper errors on data only for days of interest
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
        
        start_day = ck_lib.date_after(self.days[0])
        end_day = ck_lib.date_after(self.days[-1])
        
        dayspan = self.days[-1] - self.days[0] + 1 # time in days spanned by batch
        days_used = len(self.days) # how many days we actually have data for 
        
        # longitude ticks
        longticks = np.linspace(0,2*np.pi,9)
        longticks = [np.round(l,1) for l in longticks]
        longtick_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", 
                           r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$",
                           r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$",
                           r"$2\pi$"]
        
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
        phi = np.linspace(0, 2*np.pi, len(self.minima))
        # because of differing defns of phi=0, shift the arrays before plotting
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
        
        # =for 6-slice data: marker="s", color="#fd5956"
        # for 8-slice data: marker="o", color="#40a368"

counter = 0
for b in batches:
    counter += 1
    res = MCMC_batch(results, results_16per, results_84per, b)
    res.plot_minima()
    plt.savefig("batch"+str(counter)+"_"+str(res.ndim)+"slice.pdf")
    #plt.savefig("batch"+str(counter)+"_eck_"+str(res.ndim)+"slice.pdf")
    plt.close()
        
        
        
        
        
    
        

            
            
    
    
