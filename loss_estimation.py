# -*- coding: utf-8 -*-
"""
Created on Thu Jun 6 16:23:47 2019

@author: bjkmo
"""


import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#from scipy.signal import resample as resample
#from sklearn.metrics import mean_squared_error as mse


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
import seaborn as sns
sns.set_style('white')


# implementation of now deprecated scipy.stats.signaltonoise
# documentation can be found in historical documentation for scipy v 0.14.0
# https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py#L1864
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


#------------------------------------------------------------------------------

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
bottom=[]
depth=[]
height=[]
cond=[]
amp=[]
SNR=[]
# percentile of signal from layer file. 
# Higher percentile gives a higher bottom signal, but less reliable. 50 is median.
percent=75

 # Furesø - day 1   
for filenum in range(1,20):
    idx=filenum-1
    gain = F.Gain[idx]
    
    
    if len(str(filenum)) == 1:
        filenum='0'+str(filenum)
        
    filename = 'F_'+str(filenum)+'_amp.csv'
    filepath = os.path.join(layerdir, filename)
    
    try:
        # read layer report for scan
        da = pd.read_csv(filepath, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        # calculate signal to noise ratio
        SNR = np.append(SNR, da.Amplitude)
        # determine bottom signal amplitude. Some upper percentile of the amplitude 
        A_bottom_gained = np.percentile(da.Amplitude, percent)
        # remove gain from signal
        A_bottom = A_bottom_gained /( 10**(gain/20)   )
        # calculate loss in dB. GPRSoft uses bits with a max value of 32767
        loss = 20*np.log10(A_bottom/32767)
        # save loss to array
        losses=np.append(losses, loss)
        # save profile number to array
        profiles=np.append(profiles, 'F_'+str(filenum))
        # save bottom material, depth, and flying height to arrays
        bottom = np.append(bottom, F.Bottom[int(idx)])
        depth  = np.append(depth,  F.Depth[int(idx)])
        height = np.append(height, F.Height[int(idx)])
        cond   = np.append(cond,   F.Conductivity[int(idx)])
        amp    = np.append(amp,    A_bottom)
        
    except:
        pass

# Furesø - day 2
for filenum in range(0,11):
    idx=filenum-1
    # different gain setting on machine
    gain = F2['Gain.2'][idx]


    if len(str(filenum)) == 1:
        filenum='0'+str(filenum)
    filename = 'F2_'+str(filenum)+'_amp.csv'
    filepath = os.path.join(layerdir, filename)
    
        
        
    try:
        # read layer report for scan
        da = pd.read_csv(filepath, sep=',', skiprows=3, index_col=0, names=['Amplitude','Depth'] )
        # determine bottom signal amplitude. Some upper percentile of the amplitude 
        A_bottom_gained = np.percentile(da.Amplitude, percent)
        # remove gain from signal
        A_bottom = A_bottom_gained /( 10**(gain/20)   )
        # calculate loss in dB. GPRSoft uses bits with a max value of 32767
        loss = 20*np.log10(A_bottom/32767)
        # save profile number to array
        losses=np.append(losses, loss)
        # save profile number to array
        profiles=np.append(profiles, 'F2_'+str(filenum))
        # save bottom material, depth, and flying height to arrays
        bottom = np.append(bottom, F2.Bottom[int(idx)])
        depth  = np.append(depth,  F2.Depth[int(idx)]/100)
        height = np.append(height, F2.Height[int(idx)]/100)
        cond   = np.append(cond,   F2.Conductivity[int(idx)])
        amp    = np.append(amp,    A_bottom)

    except:
        pass


    
#---------------------------------------------------------------------------  
# All the necessary data in SI units. Losses in dB
data = pd.DataFrame(np.vstack(( depth, height, cond/10000, np.round(losses,2) ) ).astype('float') ,columns=(profiles), index=( 'depth', 'height', 'cond', 'losses')).T
data['bottom']=bottom

#------------------------ Theoretical losses --------------------------------
# Functions and constants related to theoretical amplitude losses.
# See EM calculations for details


# Velocity of EM waves in free space
c = 3e8                 # m/s
# Permittivity of free space
eps_0 = 8.89e-12        # F/m
# Dielectric constant - relative permitivitty (RP) of water
eps_water = 81          # -
# Magnetic permeability of free space
mu_0 = 1.26e-6          # H/m
# Center frequency of device
f_c = 390e6             # Hz
# Angular Frequency
omega = 2 * pi * f_c    # Hz
# Conductivity (representative value for Furesø)
sigma = 0.05            # S/m
sigma_air = 0

# RP of metal
eps_metal = np.inf

# Impedance 
eta_water = sqrt((mu_0)/(eps_water*eps_0))
eta_0 = sqrt(mu_0/eps_0)

v_water = c/sqrt(eps_water)

#---------------------------- Transmission losses ---------------------------
# function for transmission loss
transmission_loss = lambda eta_1, eta_2: 10*log10( abs( 2*eta_2/(eta_1+eta_2) )**2 *(eta_1/eta_2.conjugate()).real)  # dB
# two way loss for water
alpha_t = 2*transmission_loss(eta_water, eta_0) # dB

# ------------------------ Propagation loss -----------------------------------
# attenuation coefficient, NOT in dB
alpha_water = -sqrt(mu_0/(eps_water*eps_0)) * sigma /2 
# function for propagation loss
propagation_loss_1way = lambda d, alpha=alpha_water: 10 * log10(exp(2*alpha*d))
# simplified function for water
alpha_p = lambda conductivity, depth: -181.7 * conductivity * depth*2 # dB

#------------------------ Reflection loss -------------------------------
# function for reflection loss
reflection_loss = lambda eta_1, eta_2: 10*log10( abs( (eta_1-eta_2)/(eta_1+eta_2) )**2)
# function for reflection loss at stream bottom. Only argument is soil relative permittivity
def alpha_r(RP_soil):
  eta_soil = sqrt((mu_0)/(RP_soil*eps_0))
  return reflection_loss(eta_water, eta_soil) # dB

#--------------------- Spreading loss -------------------------------------
# spreading loss is calculated from the flying height for flying tests and for the depth for non-flying tests
lambda_c_air = c/f_c
lambda_c_water = v_water/f_c
# function for footprint area
footprint = lambda d, wavelength=lambda_c_air, eps_r = 1: (wavelength/4 + d/sqrt(eps_r + 1))**2 * 0.5 * pi
#footprint = lambda h, d, wavelength=lambda_c_air, eps_r = 1: (wavelength/4 + h/sqrt(eps_r + 1) + d/sqrt(81 + 1))**2 * 0.5 * pi

# function for spreading loss from flying height
alpha_s = lambda height, wavelength=lambda_c_air: 10*log10(footprint(0)/footprint(height,wavelength)) # dB
#alpha_s = lambda height, depth: 10*log10(footprint(0,0)/footprint(height, depth)) # dB

#--------------------- Total loss ------------------------------
# function for total loss
alpha_total = lambda RP_bottom, sigma_water, d, h: alpha_t + alpha_r(RP_bottom) + alpha_p(sigma_water, d) + alpha_s(h)  # dB
amplitude_received = lambda amplitude_transmitted, alpha_total: amplitude_transmitted * 10**(alpha_total/20)


#------------------ Calculate all losses for all profiles ----------------------

data['distance'] = data.height+data.depth
RP_dict = {'Sand':25, 'Soft':40, 'Metal':1e9 }
data['RP'] = data.bottom.map(RP_dict)
data['a_t'] = 0
data.loc[data.height>0,'a_t'] = alpha_t
data['flying'] = 'no'
data.loc[data.height>0,'flying'] = 'yes'
data['a_p'] = alpha_p(data.cond+0.033, data.depth)
data['a_s'] = alpha_s(data.height)
data.loc[data.flying=='no','a_s'] = alpha_s(data.depth, lambda_c_water)
data['a_r'] = alpha_r(data.RP)
data['a_total'] = data.a_t + data.a_p + data.a_s + data.a_r

data['diff'] = data.losses - data.a_total

#----------------- Plot measured and theoretical losses ------------------------

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(data.a_total,'-v', label='Theoretical loss')
plt.plot(data.losses,'-o', label='Measured loss')
plt.legend()
plt.xticks(rotation=45)
plt.subplot(2,1,2)
plt.plot(data.a_total-data.losses, label='Difference')
plt.twinx()
plt.plot(data.depth+data.height, 'r', label='Distance')

plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(data.a_total,'-v', label='Theoretical loss')
plt.plot(data.losses,'-o', label='Measured loss')
plt.legend()
plt.ylabel('Total loss [dB]')
plt.xticks(rotation=45)
plt.savefig('total_loss_theoretical_and_measured.pdf')
plt.show()



# ---------------------- Boxplot --------------------------------


plt.figure(figsize=(2,3))
sns.boxplot(x='flying', y='diff', data=data)
plt.xlabel('"Flying" test')
plt.ylabel('Difference \nobservation minus model [dB]')
plt.savefig('boxplot_diff_flying.pdf', bbox_inches='tight')
plt.show()



#---------------------- Plot of signal amplitude -----------------------

am = np.array(da.Amplitude)
plt.figure(figsize=(6,3))
plt.plot(am/( 10**(gain/20) ),'-k',alpha=0.7, linewidth=0.5)
plt.plot(np.arange(len(am)), np.ones(len(am))*np.mean(am)/( 10**(gain/20) ), '--r' ,label='Median')
plt.axhline(color='black', linewidth=1)
plt.legend(loc='lower right')
plt.xlabel('Trace number')
plt.ylabel('Amplitude [bits]')
plt.savefig('amlitude_of_signal.pdf')
plt.show()




#-------------------------- Scatterplots (not used in report) ---------------------------------
# 
#pd.plotting.scatter_matrix(data.loc[:,['diff', 'depth','height']])
#
#pd.plotting.scatter_matrix(data.loc[:,['diff', 'a_t', 'a_p', 'a_s', 'a_r']])


#plt.figure()
#sns.lmplot(x='distance', y='diff', data=data, ci=95, legend=True)
#plt.xlabel('Total two way travel distance [m]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()
#
#plt.figure()
#sns.lmplot(x='distance', y='diff', data=data, ci=95, legend=True, hue='flying')
#plt.xlabel('Total two way travel distance [m]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()


#plt.figure()
#sns.lmplot(x='a_s', y='diff', data=data, ci=95, legend=True, hue='flying')
#plt.xlabel('Spreading loss [dB]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()
#
#plt.figure()
#sns.lmplot(x='a_s', y='diff', data=data, ci=95, legend=True)
#plt.xlabel('Spreading loss [dB]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()
#
#
#
#plt.figure()
#sns.lmplot(x='a_p', y='diff', data=data, ci=95, legend=True, hue='flying')
#plt.xlabel('Propagation loss [dB]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()
#
#plt.figure()
#sns.lmplot(x='a_p', y='diff', data=data, ci=95, legend=True)
#plt.xlabel('Propagation loss [dB]')
#plt.ylabel('Difference \nobservation minus model [dB]')
##plt.axhline(color='k', linewidth=1)
#plt.show()


