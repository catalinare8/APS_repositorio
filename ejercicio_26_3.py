# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 21:03:22 2025

@author: Catalina
"""

#hacer con frec 1, 1.5, 250, 499

#%% módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

tt, xx = mi_funcion_sen(1.4, 0, 1, 0, 1000, 1000)
tt, xx2 = mi_funcion_sen(1.4, 0, 1.5, 0, 1000, 1000)
tt, xx3 = mi_funcion_sen(1.4, 0, 250, 0, 1000, 1000)
tt, xx4 = mi_funcion_sen(1.4, 0, 499, 0, 1000, 1000)


#%% Datos de la simulación

fs =  1000 # frecuencia de muestreo (Hz)
N =  1000 # cantidad de muestras
# con 1000 para cada una normalizamos la resolucion espectral

ts =  1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

#%% Graficos

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, xx, label= 'f=1')
plt.plot(tt, xx2, label='f=1.5')
plt.plot(tt, xx3, label= 'f=250')
plt.plot(tt, xx4, label= 'f=499')

plt.title('Señal temporal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

###########
# Espectro
###########

plt.figure(2)
ft_xx = 1/N*np.fft.fft(xx)
ft_xx2 = 1/N*np.fft.fft(xx2)
ft_xx3 = 1/N*np.fft.fft(xx3)
ft_xx4 = 1/N*np.fft.fft(xx4)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx[bfrec])**2), label='f=1' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx2[bfrec])**2), label='f=1.5' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx3[bfrec])**2), label='f=250' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx4[bfrec])**2), label='f=499' )

plt.title('Espectro')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#%% Agregado de 27/03

from scipy import signal

#hacer una funcion nueva
tt, xx = mi_funcion_sen(1.4, 0, N/4, 0, 1000, 1000)

w = signal.windows.bartlett(N)
xxw = xx * w
#normalizo
xxw1 = xxw/np.std(xxw)

plt.figure (3)
plt.plot (ff[bfrec], xxw1, label='señal con ventana normalizada' )







