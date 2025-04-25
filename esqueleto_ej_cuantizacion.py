#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: mariano
"""

#%% módulos y funciones a importar

import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> origin/main

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo
    
    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)
    
    return tt, xx

##normalizar para que la potencia sea 1
##uno es viendo la varianza pero no 
<<<<<<< HEAD
tt, xx = mi_funcion_sen(1.4, 0, 1, 0, 1000, 1000)
##print (np.var(xx))
##con desvio estandar:
xn=xx/np.std(xx)
## espectro senoidal en 0dB, cualquier cosa que veamos por debajo de eso es facil de identificar

#PARA PROBAR
# plt.figure(1)
# plt.plot (tt, xn)
# plt.title ('Señal limpia')
# plt.xlabel ('tiempo [segundos]')
# plt.ylabel ('Señal')
# plt.grid ()
# plt.show ()
=======
##tt, xx = mi_funcion_sen(1.4, 0, 1, 0, 1000, 1000)
##print (np.var(xx))
##con desvio estandar:
xn=xx/np.std(xx)
>>>>>>> origin/main

#%% Datos de la simulación

fs =  1000 # frecuencia de muestreo (Hz)
N =  1000 # cantidad de muestras
# con 1000 para cada una normalizamos la resolucion espectral

<<<<<<< HEAD
# Datos del ADC ruido digital
B =  8 # bits (los elegimos entre todos)
Vf = 1.5 # rango simétrico de +/- Vf Volts (Al graficar, para que la funcion se vea bien y no toque los bordes)
q = Vf/(2**(B-1)) # paso de cuantización de q Volts (q=Vf/2^B-1)

##1 de ganancia, fijarte el ancho de banda, y la potencia del radio 50 al cuadrado
# datos del ruido (potencia de la señal normalizada, es decir 1 W)
## tiene que ser uniforme
pot_ruido_cuant =  q**2/12 # Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

##ts =  # tiempo de muestreo
df =  fs/N # resolución espectral
=======
# Datos del ADC
B =  8 # bits (los elegimos entre todos)
Vf = # rango simétrico de +/- Vf Volts 
q =  # paso de cuantización de q Volts

##1 de ganancia, fijarte el ancho de banda, y la potencia del radio 50 al cuadrado
# datos del ruido (potencia de la señal normalizada, es decir 1 W)
xr=np.random.normal
pot_ruido_cuant =  # Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

ts =  # tiempo de muestreo
df =  # resolución espectral
>>>>>>> origin/main


#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform


# Señales

<<<<<<< HEAD
analog_sig = xx/np.std(xx)# señal analógica sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N) # señal de ruido de analógico

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)

#PARA PROBAR
# plt.figure(2)
# plt.plot (tt, sr)
# plt.title ('Señal analogica de entrada al ADC')
# plt.xlabel ('tiempo [segundos]')
# plt.ylabel ('Amplitud')
# plt.grid ()
# plt.show ()

##divido por q, redondeo y despues multiplico por q
srq = np.round(sr/q)*q # señal cuantizada

nq =  srq - sr# señal de ruido de cuantización

#PARA PROBAR
# plt.figure(3)
# plt.plot (tt, nq)
# plt.title ('Señal de ruido de cuantizacion')
# plt.xlabel ('tiempo [segundos]')
# plt.ylabel ('Amplitud')
# plt.grid ()
# plt.show ()
=======
analog_sig = # señal analógica sin ruido
sr = # señal analógica de entrada al ADC (con ruido analógico)
srq = # señal cuantizada

nn =  # señal de ruido de analógico
nq =  # señal de ruido de cuantización



>>>>>>> origin/main

#%% Visualización de resultados

# # cierro ventanas anteriores
<<<<<<< HEAD
#plt.close('all')
=======
# plt.close('all')
>>>>>>> origin/main

# ##################
# # Señal temporal
# ##################

# plt.figure(1)


# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
# plt.xlabel('tiempo [segundos]')
# plt.ylabel('Amplitud [V]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.show()

# #%% 

# plt.figure(2)
<<<<<<< HEAD
=======
#calcular espectro:
>>>>>>> origin/main
# ft_SR = 1/N*np.fft.fft( sr, axis = 0 )
# ft_Srq = 1/N*np.fft.fft( srq, axis = 0 )
# ft_As = 1/N*np.fft.fft( analog_sig, axis = 0)
# ft_Nq = 1/N*np.fft.fft( nq, axis = 0 )
# ft_Nn = 1/N*np.fft.fft( nn, axis = 0 )

<<<<<<< HEAD
# # # grilla de sampleo frecuencial
# ff = np.linspace(0, (N-1)*df, N)

# #nos quedamos con las frecuencias que esten por debajo de ff
# #bfrec es una clase de filtro
=======
# # grilla de sampleo frecuencial
# ff = np.linspace(0, (N-1)*df, N)

>>>>>>> origin/main
# bfrec = ff <= fs/2

# Nnq_mean = np.mean(np.abs(ft_Nq)**2)
# nNn_mean = np.mean(np.abs(ft_Nn)**2)

<<<<<<< HEAD
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_SR)[bfrec]**2)), ':g', label='$ s_R = s + n $' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2)[bfrec]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2)[bfrec]), ':r')
# plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2)[bfrec]), ':c')
=======
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
# plt.plot( np.array([ ff_os[bfrec][0], ff_os[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_SR)**2, axis=1)[bfrec]), ':g', label='$ s_R = s + n $' )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# plt.plot( np.array([ ff_os[bfrec][0], ff_os[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec]), ':r')
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec]), ':c')
>>>>>>> origin/main
# plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
# plt.ylabel('Densidad de Potencia [dB]')
# plt.xlabel('Frecuencia [Hz]')
# axes_hdl = plt.gca()
# axes_hdl.legend()

<<<<<<< HEAD
# # #############
# # # Histograma
# # #############

# #distribucion uniforme?
# plt.figure(4)
# bins = 10
# plt.hist(nq.flatten(), bins=bins)
# #plt.hist(nqf.flatten()/(q/2), bins=2*bins)
# plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
# #plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
# plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

# plt.xlabel('Pasos de cuantización (q) [V]')

#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='black', marker='x', ls='dotted', label='$ s $ (analog)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

#hacer mas grande una caja --> aumentar la cantidad de muestras N
#si aumento la cantidad de muestras --> aspectro mas fino, con mas resolucion
#y se comprime la sinc
#multiplicar una x por una ventana --> nuestro espectro de Xv = X(k)convolucion circular con V(k) X(k) es una delta
#


=======
# #############
# # Histograma
# #############

#distribucion uniforme?
# plt.figure(3)
# bins = 10
# # plt.hist(nq.flatten(), bins=2*bins)
# plt.hist(nqf.flatten()/(q/2), bins=2*bins)
# # plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
# plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
# plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

# plt.xlabel('Pasos de cuantización (q) [V]')
>>>>>>> origin/main
