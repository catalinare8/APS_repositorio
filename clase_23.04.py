# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:04:22 2025

@author: Catalina
"""

# probar N, N/2, N/4
# mas espectros --> bajar la varianza, podria bajar el piso
#bajar la resolucion espectral --> que no tenga pico, que este mas desparramado
#welch
#señal de ruido 20dB

#%% Declaracion

import numpy as np
from spicy import signal
import matplotlib.pyplot as plt
from scipy.signal import get_window

def autocorr(xx, max_lag):
    N=len(xx)
    result=np.correlate(xx,xx,mode='full')/N
    mid=len(result)//2 
    return result[mid:mid+max_lag]

#%% Datos

Np=1000
SNR=20 #señal de ruido
R=200 # realizaciones
fs= 1000 #frecuencia de muestreo (Hz)
N=1000 #cantidad de muestras
N2=10*N
ts= 1/fs #tiempo de muestreo
df= fs/N #resolucion espectral
df_pp= fs/N2 
a2=np.sqrt(2) #amplitud
omega0= fs/4

#%% Generacion de señal

#grilla de sampleo temporal --> discretizacion del tiempo (muestreo)
tt=np.linspace(0, (N-1)*ts, N).reshape((1000,1)) #[1000x1]
tt=np.tile(tt, (1,R)) #repetidor [100x200]

#grilla de sampleo frecuencial
ff=np.linspace(0, (N-1)*df, N) #reshape(1,1000) #[1,1000]
fr=np.random.uniform(-1/2, 1/2, size=(1,R)) #[1,200]

omega1= omega0 + fr*(df)

S=a2*np.sin(2*np.pi*omega1*tt)

#grilla de frecuencias
freqs=np.fft.fftfreq(N, d=ts)

#señal analogica --> de SNR
pot_ruido_analog=10**(-SNR/10)
sigma=np.sqrt(pot_ruido_analog)

#generacion de ruido analogico
nn=np.random.normal(0, sigma, (Np,R))

#señal final
xx=S+nn # [1000x200]

#estimador por welch
f, pot= signal.welch (xx, fs, nperseg=N, axis=0)+

#%%blackman-tukey method

max_lag=N/2 
rxx=autocorr(xx,max_lag)

#ventaneo
window=get_window('blackman', max_lag)
rxx_win=rxx*window


