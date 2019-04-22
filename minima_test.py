#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:16:10 2019

@author: nvieira
"""

import matplotlib.pyplot as plt
import cloud_killer as ck

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
batches.append([335,337,339]) # this batch crashes the kernel when making an 
# eckert projection for a 6-slice map for unknown reasons 
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

counter = 0
for b in batches:
    counter += 1
    res = ck.MCMC_batch(results, results_16per, results_84per, b)
    res.plot_minima()
    plt.savefig("batch"+str(counter)+"_"+str(res.ndim)+"slice.pdf")
    #plt.savefig("batch"+str(counter)+"_eck_"+str(res.ndim)+"slice.pdf")
    plt.close()
        