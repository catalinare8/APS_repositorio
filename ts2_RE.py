# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:46:43 2025

@author: Catalina Re
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Circuito a

#Datos
R= 1 #resistencia
L= 1 #inductor
C= 1 #capacitor

#funcion de transferencia
n=[R*C, 0]
d=[1, C*R, C*L]

#Laplace
sistema=sig.TransferFunction(n, d)

w0=1/np.sqrt(L*C)
Q=(w0*L)/R

#ancho de banda
BW=w0/Q
frec_alta=w0-(BW/2)
frec_baja=w0+(BW/2)

#angulo de fase 
ang=np.arctan(1/Q)

#funcion logaritmica con frecuencia espaciadas
ff=np.logspace(-1,3,1000) #frecuencias de 0.1 a 1000 Hz

#respuesta al sistema 
w, mag, fase = sig.bode(sistema, ff)

#Asintotas (pendientes de modulo 20dB/decada)
as_af=-20*np.log10(ff)+20*np.log10(R/L)
fase_af=90 #fase en alta frec

as_bf=20*np.log10(ff)
fase_bf=-90 #fase en baja frec


#%% Circuito b

#funcion de transferencia
n2=[1,0,0]
d2=[1,1/(R*C), 1/(L*C)]

#Laplace
sistema2=sig.TransferFunction(n2, d2)

w02=1/np.sqrt(L*C)
Q2=(w0*L)/R

#ancho de banda
BW2=w02/Q2
frec_alta2=w02-(BW2/2)
frec_baja2=w02+(BW2/2)

#angulo de fase 
ang2=np.arctan(1/Q2)

#funcion logaritmica con frecuencia espaciadas
ff2=np.logspace(-1,3,1000) #frecuencias de 0.1 a 1000 Hz

#respuesta al sistema 
w2, mag2, fase2 = sig.bode(sistema2, ff2)

#Asintotas 
as_af2=20*np.log10(R/L)*np.ones_like(ff2)
fase_af2=180 #fase en alta frec

as_bf2=40*np.log10(1/(R*C)*ff2)
fase_bf2=0 #fase en baja frec

#%% Graficos

fig, (ax1, ax2)=plt.subplots(2,1)

ax1.semilogx( w, mag, label='Módulo H(jw) [dB]')
ax1.semilogx(ff, as_bf, '--', label='Asíntota baja frecuencia')
ax1.semilogx(ff, as_af, '--', label='Asíntota alta frecuecia')
ax1.axvline(frec_baja, color='r', linestyle='--', label='Frecuencia baja BW')
ax1.axvline(frec_alta, color='r', linestyle='--', label='Frecuencia alta BW')
ax1.set_xlabel("Frecuencia [rad/seg]")
ax1.set_ylabel("Módulo [dB]")
ax1.set_xlim(1e-1, 1e3)
ax1.set_ylim(-50,10)
ax1.grid(True)
ax1.legend()

ax2.semilogx(w,fase, label='Fase H(jw) [°]')
ax2.axhline(fase_bf, color='b', linestyle='--', label='Asíntota baja frecuencia')
ax2.axhline(fase_af, color='b', linestyle='--', label='Asíntota alta frecuencia')
ax2.set_xlabel("Frecuencia [rad/seg]")
ax2.set_ylabel("Fase [°]")
ax2.grid(True)

fig2, (ax1, ax2)=plt.subplots(2,1)

ax1.semilogx( w2, mag2, label='Módulo H(jw) [dB]')
ax1.semilogx(ff2, as_bf2, '--', label='Asíntota baja frecuencia')
ax1.semilogx(ff2, as_af2, '--', label='Asíntota alta frecuecia')
ax1.axvline(frec_baja2, color='r', linestyle='--', label='Frecuencia baja BW')
ax1.axvline(frec_alta2, color='r', linestyle='--', label='Frecuencia alta BW')
ax1.set_xlabel("Frecuencia [rad/seg]")
ax1.set_ylabel("Módulo [dB]")
ax1.set_xlim(1e-1, 1e3)
ax1.set_ylim(-50,10)
ax1.grid(True)
ax1.legend()

ax2.semilogx(w2,fase2, label='Fase H(jw) [°]')
ax2.axhline(fase_bf2, color='b', linestyle='--', label='Asíntota baja frecuencia')
ax2.axhline(fase_af2, color='b', linestyle='--', label='Asíntota alta frecuencia')
ax2.set_xlabel("Frecuencia [rad/seg]")
ax2.set_ylabel("Fase [°]")
ax2.grid(True)



