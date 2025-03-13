# -*- coding: utf-8 -*-

##funcion que genere señales senoidales y permita parametrizar:
##  - la amplitud maxima de la senoidal (volts)
##  - su valor medio (volts)
##  - la frecuencia (Hz)
##  - la fase (radianes)
##  - la cantidad de muestras digitalizadas por el ADC (#muestras)
##  - la frecuencia de muestreo del ADC
##tt,xx = mi_funcion_sen (vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs)
##xx y tt deben ser vectores de Nx1

import numpy as np
import matplotlib.pyplot as plt

#%% FUNCION 

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

#%% LLAMAR A LA FUNCION
##ff=1
tt, xx = mi_funcion_sen (1, 0, 1, 0, 1000, 1000)
##ff=500
tt1, xx1 = mi_funcion_sen(1, 0, 500, 0, 1000, 1000)
##ff=999
tt2, xx2 = mi_funcion_sen(1, 0, 999, 0, 1000, 1000)
##ff=1001
tt3, xx3 = mi_funcion_sen(1, 0, 1001, 0, 1000, 1000)
##ff=2001
tt4, xx4= mi_funcion_sen(1, 0, 2001, 0, 1000, 1000)

#%% GRAFICOS

plt.figure(1)
plt.plot (tt, xx)
plt.title ('Señal Senoidal con f=1')
plt.xlabel ('tiempo [segundos]')
plt.ylabel ('Amplitud [volts]')
plt.grid ()
plt.show ()

plt.figure(2)
plt.plot (tt1, xx1)
plt.title ('Señal Senoidal con f=500')
plt.xlabel ('tiempo [segundos]')
plt.ylabel ('Amplitud [volts]')
plt.grid ()
plt.show ()

plt.figure(3)
plt.plot (tt2, xx2)
plt.title ('Señal Senoidal con f=999')
plt.xlabel ('tiempo [segundos]')
plt.ylabel ('Amplitud [volts]')
plt.grid ()
plt.show ()

plt.figure(4)
plt.plot (tt3, xx3)
plt.title ('Señal Senoidal con f=1001')
plt.xlabel ('tiempo [segundos]')
plt.ylabel ('Amplitud [volts]')
plt.grid ()
plt.show ()

plt.figure(5)
plt.plot (tt4, xx4)
plt.title ('Señal Senoidal con f=2001')
plt.xlabel ('tiempo [segundos]')
plt.ylabel ('Amplitud [volts]')
plt.grid ()
plt.show ()
