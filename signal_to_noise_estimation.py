# -*- coding: utf-8 -*-
"""
Created on Sat Jun 8 14:05:20 2019

@author: bjkmo
"""


import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#from scipy.signal import resample as resample
#from sklearn.metrics import mean_squared_error as mse

# implementation of now deprecated scipy.stats.signaltonoise
# documentation can be found in historical documentation for scipy v 0.14.0
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py#L1864
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

import numpy as np
from numpy import pi, exp, log10, sqrt
#from numpy import log as ln
#import scipy as sp
#from scipy.interpolate import interp1d as interpolate

#from datetime import date
#from datetime import datetime

#visualize SQL database 

import os

#import sqlite3

# just for graphics
try:
    import seaborn as sns
    sns.set_style('white')
except:
    pass


#------------------------------------------------------------------

# directory of GPRSoft layer files
layerdir = r'C:\Users\bjkmo\Documents\Geoscanners\GPRSoft\Layer_rep'

# path to survey notes
excelpath = r'C:\Users\bjkmo\Desktop\Bachelor_2019\GPR_field_data\GPR\surveys\2019_04_10\logBook_Mollea_and_fureso.xlsx'

# notes from first survey at Furesø 
F = pd.read_excel(excelpath, sheet_name='Fureso' , skiprows=2, index_col=0  )
F = F[F.index.notnull()]
# notes from second survey at Furesø
F2 = pd.read_excel(excelpath, sheet_name='Fureso - day 2' , skiprows=1, index_col=0  )
F2 = F2[F2.index.notnull()]

#------------------- determine losses for different scans  ------------------

losses=[]
profiles=[]
signals=[]
bottom=[]
depth=[]
height=[]
cond=[]
amp=[]
SNR=[]
noises=[]

# percentile of signal from layer file. 
# Higher percentile gives a higher bottom signal, but less reliable. 50 is median.
signal_percent=50
noise_percent=50
#---------------------------- Furesø - day 1 -------------------------------
 
files=[]

for filenum in range(1,20):
    idx=filenum-1
    gain = F.Gain[idx]
    
    
    if len(str(filenum)) == 1:
        filenum='0'+str(filenum)
        
    filename_sig = 'F_'+str(filenum)+'_amp.csv'
    filepath_sig = os.path.join(layerdir, filename_sig)
    filename_noi = 'F_'+str(filenum)+'_noise.csv'
    filepath_noi = os.path.join(layerdir, filename_noi)     
    
    try:
        # read layer report for scan
        da = pd.read_csv(filepath_sig, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        # calculate signal to noise ratio from amplitude
        S = da.Amplitude/( 10**(gain/20)   )
        SNR = np.append(SNR, signaltonoise(S))
        # determine bottom signal amplitude. Some upper percentile of the amplitude 
        A_bottom_gained = np.percentile(da.Amplitude, signal_percent)
        # remove gain from signal
        A_bottom = A_bottom_gained /( 10**(gain/20)   )
        # calculate loss in dB. GPRSoft uses bits with a max value of 32767
        loss = 20*np.log10(A_bottom/32767)
        # save loss to array
        losses=np.append(losses, loss)
        # save profile number to array
        profiles=np.append(profiles, 'F_'+str(filenum))
        # save bottom material, depth, and flying height to arrays
        signals= np.append(signals, A_bottom)
        bottom = np.append(bottom, F.Bottom[int(idx)])
        depth  = np.append(depth,  F.Depth[int(idx)])
        height = np.append(height, F.Height[int(idx)])
        cond   = np.append(cond,   F.Conductivity[int(idx)])
        
        # read layer report for noise scan
        dan = pd.read_csv(filepath_noi, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        N = dan.Amplitude/( 10**(gain/20)   )
        # determine noise amplitude. Some percentile of the absolute amplitude. 
        N_bottom_gained = np.percentile(abs(dan.Amplitude), noise_percent)
        # remove gain from noise
        N_bottom = N_bottom_gained /( 10**(gain/20)   )
        # save noise
        noises=np.append(noises, N_bottom)
        files=np.append(files, filepath_sig)
    except:
        pass
    

#----------------------------- Furesø - day 2 -------------------------------------
## 
for filenum in range(0,11):
    idx=filenum-1
    # different gain setting on machine
    gain = F2['Gain.2'][idx]
    if len(str(filenum)) == 1:
        filenum='0'+str(filenum)
    filename_sig = 'F2_'+str(filenum)+'_amp.csv'
    filepath_sig = os.path.join(layerdir, filename_sig)  
    filename_noi = 'F2_'+str(filenum)+'_noise.csv'
    filepath_noi = os.path.join(layerdir, filename_noi)     

    try:
        # read layer report for scan
        da = pd.read_csv(filepath_sig, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        # calculate signal to noise ratio from amplitude
        S = da.Amplitude/( 10**(gain/20)   )
        SNR = np.append(SNR, signaltonoise(S))
        # determine bottom signal amplitude. Some upper percentile of the amplitude 
        A_bottom_gained = np.percentile(da.Amplitude, signal_percent)
        # remove gain from signal
        A_bottom = A_bottom_gained /( 10**(gain/20)   )
        # save profile
        profiles=np.append(profiles, 'F2_'+str(filenum))
        # save signal
        signals= np.append(signals, A_bottom)
        bottom = np.append(bottom, F2.Bottom[int(idx)])
        depth  = np.append(depth,  F2.Depth[int(idx)]/100)
        height = np.append(height, F2.Height[int(idx)]/100)
        cond   = np.append(cond,   F2.Conductivity[int(idx)])
        
        # read layer report for noise scan
        dan = pd.read_csv(filepath_noi, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        N = dan.Amplitude/( 10**(gain/20)   )
        # determine noise amplitude. Some percentile of the absolute amplitude. 
        N_bottom_gained = np.percentile(abs(dan.Amplitude), noise_percent)
        # remove gain from noise
        N_bottom = N_bottom_gained /( 10**(gain/20)   )
        # save noise
        noises=np.append(noises, N_bottom)
        
    except:
        pass


#-------------------------- Plots ---------------------------


plt.figure(figsize=(6,3))
plt.plot(profiles[1:], log10(signals[1:]),'x', label='Signal')
plt.plot(profiles[1:], log10(noises[1:]),'--', label='Noise')
plt.legend()
#plt.axis([-.5,len(profiles)-.5,0, 500])
plt.ylabel('$log_{10}$ Amplitude [bits]')
plt.xticks(fontsize=8,rotation=45)
plt.savefig('signal_and_noise.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(4,3))
plt.plot(profiles[15:], (signals[15:]/noises[15:]), 'xk', label='SNR')
#plt.legend()
plt.ylabel('Signal-to-noise ratio')
plt.xticks(rotation=45)
plt.ylim((0,2.65))
plt.axhline(y=1,color='black', linewidth=0.5)
plt.savefig('signal_to_noise.pdf', bbox_inches='tight')
plt.show()


#----------------- Example using signaltonoise function  ---------------------
plt.figure()
plt.plot(S,'.')
plt.plot(N,'.')
plt.show()


