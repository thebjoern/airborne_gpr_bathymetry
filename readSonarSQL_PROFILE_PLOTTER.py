# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:45:12 2016

@author: bjkmo
"""

import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#from scipy.signal import resample as resample
from sklearn.metrics import mean_squared_error as mse
from numpy import sqrt

import numpy as np
#import scipy as sp
from scipy.interpolate import interp1d as interpolate
from scipy.stats import norm, skew
#from datetime import date
from datetime import datetime

#visualize SQL database 

import os

import sqlite3

# just for graphics
try:
    import seaborn as sns
    sns.set_style('white')
except:
    pass


# removes outliers outside 1.5 the interquantile range
# modified from https://gist.github.com/vishalkuo/f4aec300cf6252ed28d3
def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    resultList= a[((lower_quartile - IQR) < a) & (a < (upper_quartile + IQR))]
    return resultList


#--------------------------------------------------------------------------------------
# path to sonar files
sonar_path = r'C:\Users\bjkmo\Desktop\Bachelor_2019\sonar_files'

# path to GPR layer files
GPR_filedir = r'C:\Users\bjkmo\Documents\Geoscanners\GPRSoft\Layer_rep'

# read GPR-sonar couples from Excel spreadsheet
data = pd.read_excel(r'C:\Users\bjkmo\Desktop\Bachelor_2019\GPR_sonar_profile_couples.xlsx').dropna()


def GPR_sonar_couple(datarow, elevation_correction=True, show_plots='both', save_plots=False):
    #---------------------------------------------------------------------------------------------------------------
    # Calculates RMSE and bias between GPR layer file (.csv) and sonar profile from Deeper Pro+ Sonar.
    # Optional: Plots profiles and difference between the two
    # Arguments:
    #   datarow: pandas data series with columns:   GPR_profile_no: GPR profile number so that 'GPR_depth_profile_' + arg + '.csv' is layer filename
    #                                               sonar_file:     Name of sonar file 
    #                                               lower:          Lower scan number at which to cut sonar profile, qualitative assessment. Might be upgraded to timestamp in later edition 
    #                                               upper:          Upper scan number at which to cut sonar profile, qualitative assessment. Might be upgraded to timestamp in later edition 
    #                                               elevation:      GPR unit elevation above water surface at time of data collection. Unit = (cm)
    #                                               profile_number: GPR profile number, only for plot title purposes
    #                                               location:       GPR profile location. So far only 'F' for Furesø, 'M' for Mølleåen, or 'U' for Usserød
    #   elevation_correction: boolean. Correct GPR data for elevation above water level
    #   show_plots: string
    #       'both':             show plots of profiles and difference between profiles
    #       'profiles':         show plot of profiles
    #       'none' or other:    do not show plots
    #   save_plots: boolean. Save plot as .png
    #
    # Outputs: 
    #   1: RMSE between GPR and sonar in cm
    #   2: Bias between GPR and sonar in cm
    #
    # The part for reading sonar files is written by Filippo Bandini, DTU ENV.
    #-----------------------------------------------------------------------------------------------------------------
    GPR_filename = 'GPR_depth_profile_'+str(datarow.GPR_profile_no)+'.csv'
    sonar_data = datarow.sonar_file
    a = int(datarow.lower)           
    b = int(datarow.upper)          
    elev = datarow.elevation         
    wstt = elev / 30 * 2 / 100          # water surface travel time: elevation(cm) / speed of light in air(cm/ns)
    
    
    file_to_read= os.path.join(sonar_path, sonar_data)
    connection = sqlite3.connect(file_to_read)
    cursor = connection.cursor()
    cursor.execute("SELECT time, depth FROM SONAR_DATA;")
    results = cursor.fetchall()
    #for r in results:
    #    print(r)
    cursor.close()
    connection.close()
    
    depth=[]
    timeUNIX=[]
    date_time=[]
    idx=0
    for row in results:
         
         
         depth.append(results[idx][1]) #depth
         timeUNIX.append(results[idx][0]) #POSIX time 
         date_time.append(datetime.fromtimestamp(results[idx][0]/1000))#from UNIX time to date-time MATLAB time_datestr_sonar=datestr(deeper_series.time(:)/86400/1000+3600*2/86400+ datenum(1970,1,1));
         idx=idx+1
    
    
    
    # plot sonar data
#    plt.plot(date_time, depth)     
    
    # read GPR data from GPRSoft layer report (.csv)
    GPR_filepath = os.path.join(GPR_filedir, GPR_filename)
    GPR_data = pd.read_csv(GPR_filepath, sep=',', skiprows=3, names=('Trace','Depth'))
    
    if elevation_correction:
    # subtract flying height from GPR depth
        GPR_data.Depth = GPR_data.Depth - wstt
    
    # cut sonar data to fit with the GPR scan. This is done qualitatively for each scan
    sonar_data_cut = depth[a:b]
    
    # resample GPR data to length of cut sonar data    
    new_length = len(sonar_data_cut)
    new_x = np.linspace(GPR_data.Trace.min(), GPR_data.Trace.max(), new_length)
    GPR_depth_interpolated = interpolate(GPR_data.Trace, GPR_data.Depth, kind='cubic')(new_x)
    
    
    #---------------------- using interpolation ----------------------------
    
    # calculate statistics and make string for plot
    RMSE = sqrt(mse(GPR_depth_interpolated, sonar_data_cut))
    bias = sum(GPR_depth_interpolated - sonar_data_cut)/len(GPR_depth_interpolated)
    textstr = '\n'.join((
        r'$\mathrm{RMSE}=%.2f$ cm' % (RMSE*100, ),
        r'$\mathrm{bias}=%.2f$ cm' % (bias*100, ))
        )
    
    
    # plot
    if show_plots.lower()=='profiles':
        titletext = 'Profile '+str(int(datarow.profile_number))+' - '+datarow.location + '\nFlying height: '+str(int(datarow.elevation))+' cm'
        # make figure
        plt.figure( figsize=(10,4))
        
        # plot GPR and sonar together
        plt.plot(GPR_depth_interpolated, label='GPR data')
        plt.plot(sonar_data_cut, 'k', label='Sonar data') 
        plt.legend(loc='lower left')
        plt.title(titletext)
        plt.xlabel('Data point')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()
        plt.gca().text(0.84, 0.97, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top') 
        if save_plots:
            plt.savefig('GPR_sonar_profile_'+datarow.GPR_profile_no+'_profile_only.png',  pad_inches=0.5,)
        
    elif show_plots.lower()=='both':
        # make string for plot title
        titletext = 'Profile '+str(int(datarow.profile_number))+' - '+datarow.location + '\nFlying height: '+str(int(datarow.elevation))+' cm'
        # make figure
        plt.subplots(2,1, figsize=(10,8))
        
        # plot GPR and sonar together
        plt.subplot(2,1,1)
        plt.plot(GPR_depth_interpolated, label='GPR data')
        plt.plot(sonar_data_cut, 'k', label='Sonar data') 
        plt.legend(loc='lower left')
        plt.title(titletext)
        plt.xlabel('Data point')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()
        plt.gca().text(0.84, 0.97, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top')
        
        # plot difference
        plt.subplot(2,1,2)
        plt.plot((GPR_depth_interpolated-sonar_data_cut),'g', label='Difference')
        plt.legend(loc='upper left')
        plt.gca().axhline(color='black', linewidth=.5)
        plt.xlabel('Data point')
        plt.ylabel('Depth (m)')
        
        if save_plots:
            plt.savefig('GPR_sonar_profile_'+datarow.GPR_profile_no+'_both.png',  pad_inches=0.5,)
        
        plt.show()
    else: 
        pass
    
    
    return RMSE*100, bias*100, GPR_depth_interpolated, sonar_data_cut
#----------------------------------------------------------------------------------------



# now use function on all 8 profiles
n=8


aa=np.zeros(n)
bb=np.zeros(n)
gpr_all=[]
sonar_all=[]
#------------------------ Plot all profiles ------------------------------
for i in range(n):
    aa[i], bb[i], gpr_data, sonar_data = GPR_sonar_couple(data.iloc[i,:], elevation_correction=True, show_plots='both',save_plots=True)
    gpr_all = np.append(gpr_all, gpr_data)
    sonar_all = np.append(sonar_all, sonar_data)


#----------------------------- Boxplot ------------------------------

df=pd.DataFrame({'RMSE':aa,'Bias':bb})
plt.figure(figsize=(2,5))
sns.boxplot(data=df)
plt.ylabel('cm')
plt.savefig('boxplot_rmse_bias.pdf', bbox_inches='tight')
plt.show()


# ---------------------- GPR/sonar scatterplot -------------------------------------
plt.figure(figsize=(5,5))
plt.plot(gpr_all*100, sonar_all*100, '.',alpha=0.1, markersize=10, label='SPG-1800, Deeper, scatter')
plt.plot((0,100), (0,100), '-k', label='1:1')
plt.axis([0,100,0,100])
plt.xlabel('GPR [cm]', fontsize=15)
plt.ylabel('Sonar [cm]', fontsize=15)
plt.legend()
plt.savefig('scatterplot_gpr_x_sonar.pdf', bbox_inches='tight')#,  pad_inches=0.5,)
plt.show()



#------------------------ GPR-sonar boxplot -----------------------------
#test removeOutliers function
b = gpr_all*100 - sonar_all*100
plt.figure()
plt.subplot(1,2,1)
plt.boxplot(b)
plt.axis([0.5,1.5,-40,15])
plt.subplot(1,2,2)
plt.boxplot(removeOutliers(b,1.5), whis='range')
plt.axis([0.5,1.5,-40,15])
plt.show()


# --------------------- GPR-sonar histogram -----------------------------------------


h = gpr_all*100 - sonar_all*100
h=removeOutliers(h,1.5)
std = np.std(h) 
mean = np.mean(h)    
x = np.linspace(min(h),max(h), 1000)
sk = skew(h)

histstr = '\n'.join((
    r'$\mathrm{mean}  =%.2f$ cm' % (mean, ),
    r'$\mathrm{std}   =%.2f$ cm' % (std, ),
    r'$\mathrm{skew}  =%.2f$' % (sk, ),
    )
    )

plt.figure(figsize=(5,5))
plt.hist(h, bins=20, density=True)
plt.plot(x, norm.pdf(x, mean, std), '-r', linewidth=3,  label='PDF of corresponding \nnormal distribution')
plt.xlabel('GPR minus sonar (cm)', fontsize=15)
plt.text(3.8,0.15, histstr)
plt.legend()
plt.savefig('histogram_gpr_minus_sonar.pdf')#,  pad_inches=0.5,)
plt.show()








# Calculates statistics for profiles with and without elevation correction
# --------------------------------------
#RMSE_old=np.zeros(n)
#bias_old=np.zeros(n)
#RMSE_new=np.zeros(n)
#bias_new=np.zeros(n)
#for i in range(n):
#    RMSE_new[i], bias_new[i] = GPR_sonar_couple(data.iloc[i,:], elevation_correction=True, show_plots=False)
#    RMSE_old[i], bias_old[i] = GPR_sonar_couple(data.iloc[i,:], elevation_correction=False, show_plots=False)
#    
#
#box_textstr='\n'.join((
#        r'$\mathbf{Means}$',
#        r'Old RMSE   $=$   $%.2f$ cm' % (RMSE_old.mean(), ),
#        r'Old bias      $=$   $%.2f$ cm' % (bias_old.mean(), ),
#        r'New RMSE  $=$   $%.2f$ cm' % (RMSE_new.mean(), ),
#        r'New bias     $=%.2f$ cm' % (bias_new.mean(), ))
#            )
#
## make boxplot showing difference between statistics with and without elevation correction
#plt.figure(figsize=(6,4))
#plt.boxplot((RMSE_old, bias_old, RMSE_new, bias_new), labels=('Old RMSE', 'Old Bias', 'New RMSE', 'New Bias'), widths = 0.7, sym='xk')
#plt.gca().text(1.05, 0.97, box_textstr, transform=plt.gca().transAxes, fontsize=10,
#               verticalalignment='top')
##plt.savefig('GPR_sonar_boxplot.png')
#plt.show()
# --------------------------------------------------------
