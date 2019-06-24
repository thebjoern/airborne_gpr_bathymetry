import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import h5py
import os
#from scipy.signal import argrelextrema as extrema
import seaborn
seaborn.set_style('white')
from numpy import pi, exp, log10, sqrt


os.chdir(r'C:\Users\bjkmo\Desktop\Bachelor_2019\outfiler')
h5dir = os.getcwd()


ds = []
scans = []
for file in os.listdir(h5dir):
    if file.endswith('out'):
        ds.append(os.path.join(h5dir, file))
        scans.append(file[-13:-11])


K=np.repeat(2000,len(ds))
K[6], K[7], K[8], K[9], K[10], K[11] =3000,3900,2500, 2500, 2500, 2500
times=np.repeat(30,12)
times[4:]=60

losses_top=0
losses_bot=0
direct_amp=[]
loss_means=[]


for n in range(len(ds) ):
    # read data from .hdf5 file
    f = h5py.File(ds[n])
    # array from all traces from 0 to 5000 its
    time = times[n]
    xx = 5000
    arr = f['rxs']['rx1']['Ez'][:xx,:]
    plottime= len(arr)/len(f['rxs']['rx1']['Ez'])*time
    its_per_ns = int(len(arr)/plottime)
    tick_loc = np.arange(0, xx, 2*its_per_ns)
    tick_marks = np.arange(0, len(tick_loc))*2
    
    
    X_top=[]
    X_top2=[]
    X_bot=[]
    X_bot2=[]
    
    Y_top=[]
    Y_top2=[]
    Y_bot=[]
    Y_bot2=[]   
    # for each trace
    n_trace = range(np.shape(f['rxs']['rx1']['Ez'])[1])
    for trace in n_trace:     #[4,5,6,7]: 
    #    plt.plot(f['rxs']['rx1']['Ez'][:,i])
        k=K[n]
        
        line = arr[:, trace]
        bline=line[k:]
        rn = np.arange(0, len(line))
        Y_top = np.append(Y_top, max(line))
        Y_bot = np.append(Y_bot, min(line))
        Y_top2 = np.append(Y_top2, max(bline))
        Y_bot2 = np.append(Y_bot2, min(bline))
      
        X_top=np.append(X_top , rn[line== max(line)])
        X_top2= np.append(X_top2 , rn[line==max(bline)])
        X_bot=np.append(X_bot , rn[line== min(line)])
        X_bot2= np.append(X_bot2 , rn[line==min(bline)])
    
    
    loss_top=20*np.log10(Y_top2/Y_top)
    loss_bot=20*np.log10(Y_bot2/Y_bot)
    
    direct_amp = np.append(direct_amp, np.mean(abs(Y_top)+abs(Y_bot))/2 )
    
    try:
        losses_top = np.vstack((losses_top, loss_top))
    except:
        losses_top =  loss_top
    
    try:
        losses_bot = np.vstack((losses_bot, loss_bot ))
    except:
        losses_bot =  loss_bot
    
    set_gain = 30
    gain_dB = np.append(np.zeros(round( len(arr) - len(arr)/1.5)), np.linspace(0, set_gain, round(len(arr)/1.5)) )
    gain = 10**(gain_dB/20)
    arr = (arr.T*gain).T
    
    # ------------------------------- plot scan ------------------------------
    plt.figure(figsize=(6,3))
#    plt.suptitle(scans[n])
    levels=np.linspace(-700, 700, 501)
    con=plt.contourf(arr, levels=levels, cmap='Greys_r')
    plt.plot(n_trace, X_top, '*k')
    plt.plot(n_trace, X_bot, '*k')
    plt.plot(n_trace, X_top2, '*k')
    plt.plot(n_trace, X_bot2, '*k')
    plt.gca().invert_yaxis()
    cbar=plt.colorbar(con, ticks=np.linspace(-700, 700, 15), label=r'Field strength [V m$^{-1}$]')
    plt.yticks(tick_loc, tick_marks)
    plt.ylabel('Two-way travel time [ns]')
    plt.xlabel('Trace number')
    cbar.ax.tick_params(labelsize=8) 
#    plt.twinx()
#    plt.plot(n_trace, loss_top)
#    plt.plot(n_trace, loss_bot)
#    plt.plot(n_trace, (loss_top+loss_bot)/2, '--')
#    plt.axis([0,len(n_trace)-1, 0,-50])
#    plt.gca().invert_yaxis()
#    plt.savefig('gprMax_model'+scans[n]+'_w_gain.png', dpi=300)
    plt.show()    


    



# ------------------------------ Loss calculations -------------------------- 

# print all losses to spreadsheet
losses = (losses_top+losses_bot)/2
df=pd.DataFrame(losses , index=scans )
#df.to_excel('Losses_from_EandF.xlsx')

df['Mean']=np.mean(df, axis=1)
df['direct']=direct_amp
df2=df.loc[['E5','E6','E7','E8']]
df=df.drop(['E5','E6','E7','E8'])


simdata = pd.read_excel('GPRMaxsimulations.xlsx')

data = pd.concat([df[['Mean', 'direct']], simdata[['Water depth', 'Elevation','RDP','Cond', 'RDP_soil' ]]  ], axis=1, join_axes=[df.index])
data.RDP = 80




#-------------- Calculations and funcitons from EM_calc.ipynb ----------------
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

# function for spreading loss from flying height and depth
alpha_s = lambda height, depth : 10*log10(0.1**2 / ((height)**2 + (depth/9**2) )) # dB

#--------------------- Total loss ------------------------------
# function for total loss
alpha_total = lambda RP_bottom, sigma_water, d, h: alpha_t + alpha_r(RP_bottom) + alpha_p(sigma_water, d) + alpha_s(h)  # dB
amplitude_received = lambda amplitude_transmitted, alpha_total: amplitude_transmitted * 10**(alpha_total/20)
#--------------------------------------------------------------------------------------------------------

data = data.astype('float')
data['a_t'] = alpha_t
data['a_p'] = alpha_p(data.Cond+0.033, data['Water depth'])
data['a_s'] = alpha_s(data.Elevation, data['Water depth'])
data['a_r'] = alpha_r(data.RDP_soil)
data['a_total'] = data.a_t + data.a_p + data.a_s + data.a_r
data['Diff'] = data.Mean - data.a_total


# ----------------------- Plot total losses ---------------------------------
plt.figure(figsize=(5,3))
plt.plot(data.Mean,'-', label='gprMax model loss')
plt.plot(data.a_total,'--', label='Calculated loss')
#plt.plot(data.a_s)
#plt.plot(data.a_t)
#plt.plot(data.a_r)
#plt.plot(data.a_p)
plt.ylim(0, -40)
plt.ylabel('Loss [dB]')
plt.xlabel('Model number')
plt.legend()
plt.savefig('gprMax_models.pdf')
plt.show()




dfmd=df2[['Mean','direct']]
df2=df2.drop(columns=['Mean','direct'])
RDP_soil=6
Cond=0
Elev=[0.25, 0.5, 1, 1.5]
df2 = df2.astype('float')
slope_depth=np.linspace(0.1, 0.2, 35)



names=['E5', 'E6', 'E7', 'E8']
for i in [0,1,2,3]:
    df2.loc['a_total_'+names[i],:] = alpha_t + alpha_r(RDP_soil) + alpha_p(Cond+0.033, slope_depth) + alpha_s(Elev[i], slope_depth)


plt.subplots(2,2, sharey='row', sharex='col')
for i in [0,1,2,3]:
    plt.subplot(2,2,i+1)
    plt.plot(df2.loc[names[i]], label='gprMax')
    plt.plot(df2.loc['a_total_'+names[i]], label='EM calc')
#plt.legend()
#plt.twinx()
#plt.plot(df3.depth, '--k')
#plt.gca().invert_yaxis()
plt.show()

