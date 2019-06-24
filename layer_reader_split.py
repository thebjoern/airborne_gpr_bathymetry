# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:13:09 2019

@author: bjkmo
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt
from sklearn.metrics import mean_squared_error as mse


b1=[]
b2=[]
b3=[]
profiles = []

# read GPR data from GPRSoft layer report (.csv)
filedir = r'C:\Users\bjkmo\Documents\Geoscanners\GPRSoft\Layer_rep'
for filenum in range(0,20):
    for bandnum in (1,2,3):
        filename = 'P'+str(filenum)+'L'+str(bandnum)+'.csv'
        filepath = os.path.join(filedir, filename)
        try:
            da = pd.read_csv(filepath, sep=',', skiprows=3 , names=['Depth'])
            if bandnum==1:
                b1 = np.append(b1, da.Depth.mean())
            if bandnum==2:
                b2 = np.append(b2, da.Depth.mean())
            if bandnum==3:
                b3 = np.append(b3, da.Depth.mean())
                profiles = np.append(profiles, 'P'+str(filenum))
            
        except:
            pass


t1,t2,t3 = b1/0.033*2, b2/0.033*2, b3/0.033*2 
offset = np.concatenate( [np.ones(7)*8, np.ones(3)*0, np.ones(5)*4])
offset_depth = offset * 0.033/2
measured = np.array([.4, .44, .44, .46, .54, .69, .85, 1.01, .78, .9, .9, .9, .8, .8, .8])
theoretical = measured /0.033*2
data = pd.DataFrame([b1,b2,b3,t1,t2,t3,offset,offset_depth,measured,theoretical]).T
data.index = profiles
data.columns = ['Band_1','Band_2','Band_3','Time_1','Time_2','Time_3','Offset','Offset_depth','Measured','Theoretical']

# save layer data to .csv file
data.to_csv(path_or_buf='Layer_data.csv',sep=';')


print(r'We see that by adjusting for the offset, the profiles with an offset of 8 ns get very close to the measured. The profiles with an offset of 0 are not close to the ground truth, and the profiles with offset 4 are more wrong when adjusted.')
print(r'The profiles with an offset of 0 and 4 ns were conducted on deeper water. There might be a higher uncertainty for the ground truth data on deeper water. They were conducted further from the shore where there is more wind, making the boat rock more.')

# compared to ground truth
fig, axs = plt.subplots(1,3,figsize=(10,3),gridspec_kw={'width_ratios': [6,2,4.5]} , sharey=True)
idx_arr=np.array([[0,7],[7,10],[10,15]])

fig.suptitle('Difference between GPR measurements and ground truth in Furesø \nGPR data is adjusted for offset')
for i in (1,2,3):

    idx=np.arange(idx_arr[i-1,0],idx_arr[i-1,1])
    
    ax=axs[i-1]

    ax.plot(data.Time_1[idx]-data.Offset[idx]-data.Theoretical[idx], 'o', label='Band 1')
    ax.plot(data.Time_2[idx]-data.Offset[idx]-data.Theoretical[idx], '^', label='Band 2')
    ax.plot(data.Time_3[idx]-data.Offset[idx]-data.Theoretical[idx], 'D', label='Band 3')
    ax.axhline(color='black', linewidth=.5)
    
    if i==1:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0*1.2, box.width, box.height*.88])
        ax.set(xlabel='Offset = 8 ns',ylabel='Difference from ground truth (ns)')
    if i==2:    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0*1.2, box.width, box.height*.88])
        ax.set(xlabel='Offset = 0 ns')
    if i==3:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0*1.2, box.width * 0.8, box.height*.88])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set(xlabel='Offset = 4 ns')    
    plt.ylim((-12,12))

plt.savefig('band_difference_plot.png')
plt.show()



# save key figures to .csv file
rmsearr=np.zeros((3,3))
biasarr=np.zeros((3,3))
for i in (0,1,2):
    idx=np.arange(idx_arr[i,0],idx_arr[i,1])
    #calculate statistics for each band
    rmsearr[i,0] = sqrt(mse(data.Time_1[idx]-data.Offset[idx], data.Theoretical[idx]))
    rmsearr[i,1] = sqrt(mse(data.Time_2[idx]-data.Offset[idx], data.Theoretical[idx]))
    rmsearr[i,2] = sqrt(mse(data.Time_3[idx]-data.Offset[idx], data.Theoretical[idx]))  
    biasarr[i,0] = sum(data.Time_1[idx]-data.Offset[idx]-data.Theoretical[idx])/len(data.Time_1[idx])
    biasarr[i,1] = sum(data.Time_2[idx]-data.Offset[idx]-data.Theoretical[idx])/len(data.Time_1[idx])
    biasarr[i,2] = sum(data.Time_3[idx]-data.Offset[idx]-data.Theoretical[idx])/len(data.Time_1[idx])

stats_data = pd.DataFrame(np.hstack((rmsearr, biasarr)), index=('Offset 8 ns','Offset 0 ns','Offset 4 ns'), columns=('Band 1','Band 2','Band 3','Band 1','Band 2','Band 3')).T
new_idx=pd.MultiIndex(levels=[['RMSE', 'Bias'], ['Band 1','Band 2','Band 3']],labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
stats_data.index=new_idx
stats_data=stats_data.T
stats_data.to_csv(path_or_buf='Layer_stats.csv',sep=';')



#------------------- with and without offset -------------------------------
#
## compared to ground truth
#fig, axs = plt.subplots(2,3,figsize=(10,6),gridspec_kw={'width_ratios': [6,2,4.5]} , sharey=True)
#idx_arr=np.array([[0,7],[7,10],[10,15]])
#
#
#for i in (1,2,3):
#    
#    
#    ax1=axs[0,i-1]
#    idx=np.arange(idx_arr[i-1,0],idx_arr[i-1,1])
###    ax.subplot(2,3,i)
###    ax.title('Not adjusted for offset')
#    ax1.plot(data.Time_1[idx]-data.Theoretical[idx], 'o', label='Band 1')
#    ax1.plot(data.Time_2[idx]-data.Theoretical[idx], '^', label='Band 2')
#    ax1.plot(data.Time_3[idx]-data.Theoretical[idx], 'D', label='Band 3')
#    ax1.axhline(color='black', linewidth=.5)
#    plt.ylim([-12,12])
#    #legend
#
#    
#    if i==3:
#        box = ax1.get_position()
#        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
##        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
##    
##    
#    
#    ax2=axs[1,i-1]
##    ax.subplot(2,3,i+3)
##    ax.title('Adjusted for offset')
#    ax2.plot(data.Time_1[idx]-data.Offset[idx]-data.Theoretical[idx], 'o', label='Band 1')
#    ax2.plot(data.Time_2[idx]-data.Offset[idx]-data.Theoretical[idx], '^', label='Band 2')
#    ax2.plot(data.Time_3[idx]-data.Offset[idx]-data.Theoretical[idx], 'D', label='Band 3')
#    ax2.axhline(color='black', linewidth=.5)
#    #legend
#    if i==1:
#        ax2.set(ylabel='Difference (ns)')
#    
#    if i==3:
#        box = ax2.get_position()
#        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#        ax2.legend(loc='center left', bbox_to_anchor=(1, 1.1))
#    
#    plt.ylim((-12,12))
#
#plt.show()









#
#filedir = r'C:\Users\bjkmo\Desktop\Bachelor_2019'
#filename = 'Layercake.csv'
#filepath = os.path.join(filedir, filename)
#
#rownames = (['P02','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13','P14','P15','P16','P17'])
#data = pd.read_csv(filepath, sep=';')
#data.index = rownames
#data[['B1','B2','B3']] = data[['Band_1', 'Band_2', 'Band_3']] - data[['Offset','Offset','Offset']].values
#data = data.dropna()
#
#plt.figure(figsize=(10,7))
#
#plt.subplot(2,1,1)
#plt.title('Not adjusted for offset')
#plt.plot(data.Band_1, 'x', label='Band 1')
#plt.plot(data.Band_2, 'x', label='Band 2')
#plt.plot(data.Band_3, 'x', label='Band 3')
#plt.plot(data.Theoretical, '^k')
##legend
#box = plt.gca().get_position()
#plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
#plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#
#plt.subplot(2,1,2)
#plt.title('Adjusted for offset')
#plt.plot(data.B1, 'x', label='Band 1')
#plt.plot(data.B2, 'x', label='Band 2')
#plt.plot(data.B3, 'x', label='Band 3')
#plt.plot(data.Theoretical, '^k')
##legend
#box = plt.gca().get_position()
#plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
#plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
#

#print(data)
