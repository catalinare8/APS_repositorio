# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 08:50:28 2025

@author: Catalina Re
"""

#Siendo s la señal de ECG registrada con interferencias, y (x)^ la señal filtrada, una estimación del ECG sin interferencias:
#   (x)^=s-(b)^

#Se pide que imlemente ambas estimaciones de b detalladas a continuación: 
#   1) Filtro de mediana
#   (puede utilizar la implementación del filtro de mediana provista en scipy.signal)
#   2) Interpolación mediante splines cúbicos
#   3) Filtro adaptado (matched filter)
#   (revise el concepto de filtro adaptado en Wikipedia, o la bibliografía de la materia)
#       a) Explique conceptualmente cómo podría realizar un detector de latidos con la señal resultante del filtro adaptado. Discuta la utilidad y limitaciones de este método
#       b) Realice la detección de los latidos, comparando las detecciones obtenidas con las que se incluyen en la variable qrs_detections. Proponga alguna métrica (sensibilidad, valor predictivo positivo) para cuantificar la performance del detector

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy.interpolate import CubicSpline
from scipy.signal import correlate, find_peaks

mat_struct=sio.loadmat('./ECG_TP4.mat')
ecg_one_lead=mat_struct['ecg_lead'].flatten()
N1=len(ecg_one_lead)

ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead) #normalizar la señal

qrs_pattern1=mat_struct['qrs_pattern1'].flatten()
heartbeat_pattern1=mat_struct['heartbeat_pattern1'].flatten()
heartbeat_pattern2=mat_struct['heartbeat_pattern2'].flatten()
qrs_detections=mat_struct['qrs_detections'].flatten()

#señal en el tiempo
plt.figure()
plt.plot(ecg_one_lead, label='Señal de ECG completa')
plt.title("ECG con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.grid(True)

senales={
    'QRS (latido normal)': qrs_pattern1,
    'Latido normal completo': heartbeat_pattern1,
    'Latido ventricular': heartbeat_pattern2,
    'Señal con detecciones QRS': qrs_detections
    }

fig, axs=plt.subplots(2,2,figsize=(12,10), sharey=False)
axs=axs.flatten()

for idx, (titulo, senal) in enumerate(senales.items()):
    axs[idx].plot(senal, label=titulo)
    axs[idx].set_title(f'{titulo}')
    axs[idx].set_xlabel('Muestras')
    axs[idx].set_ylabel('Amplitud [V]')
    axs[idx].grid(True)
    axs[idx].legend()
    
plt.tight_layout()
plt.show()

#%%Filtro
#segmento de interés
fs=1000 #Hz
ecg_segmento=ecg_one_lead[700000:745000]
t_segmento=np.linspace(0,len(ecg_one_lead)/fs,len(ecg_segmento))

#ventana
win1=201
win2=1201

#filtro de mediana (200ms)
linea_base=sig.medfilt(ecg_segmento, kernel_size=win1)

#filtro de mediana (600ms)
linea_base2=sig.medfilt(linea_base, kernel_size=win2)

x_est=ecg_segmento-linea_base2

plt.figure(figsize=(12,5))
plt.plot(t_segmento,ecg_segmento,label='Señal sin filtrar')
plt.plot(t_segmento,linea_base2,label='Linea de base por mediana')
plt.plot(t_segmento,x_est,label='Señal Filtrada')
plt.title("ECG")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

#%%Interpolación
n0=int(0.1*fs)
ecg_segmento=ecg_one_lead[700000:745000]

m_i=qrs_detections-n0
m_i=m_i[(m_i>=0)&(m_i<len(ecg_segmento))]

s_mi=ecg_segmento[m_i]

#interpolación con splines cúbicos
spline=CubicSpline(m_i, s_mi)
n=np.arange(len(ecg_segmento)) #evaluar el spline en todos los puntos
base_spline=spline(n)

#filtrar la señal
x_est2=ecg_segmento-base_spline

plt.figure()
plt.plot(ecg_segmento,label='Señal original')
plt.plot(base_spline, label='Línea de base por Splines')
plt.plot(x_est2,label='Señal filtrada')
plt.legend()
plt.title("ECG y línea de base estimada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#%%Matched Filter
ecg_segmento=ecg_one_lead[300000:312000]
h=qrs_pattern1[::-1]

correlation=correlate(ecg_segmento,h,mode='same')

peaks, _ = find_peaks(correlation, height=np.max(correlation)*0.3, distance=int(fs*0.6))

peaks_true=qrs_detections[(qrs_detections>=300000)&(qrs_detections<312000)]-300000

fig,axs=plt.subplots(2,1,figsize=(12,8))

axs[0].plot(ecg_segmento, label='ECG')
axs[0].plot(peaks_true, ecg_segmento[peaks_true], 'go', label='Latidos reales')
axs[0].plot(peaks,ecg_segmento[peaks], 'rx', label='Latidos detectados')
axs[0].set_title('Señal ECG (latidos reales)')
axs[0].set_xlabel('Muestras')
axs[0].set_ylabel('Amplitud')
axs[0].grid(True)
axs[0].legend()

#correlación con detección de picos
axs[1].plot(correlation,label='Correlación (matched filter)')
axs[1].plot(peaks_true, correlation[peaks_true], 'go', label='Latidos reales')
axs[1].plot(peaks, correlation[peaks], 'ro', label='Latidos detectados')
axs[1].set_title('Filtro adaptado - correlación y detecciones')
axs[1].set_xlabel('Muestras')
axs[1].set_ylabel('Amplitud')
axs[1].grid(True)
axs[1].legend()
plt.tight_layout()
plt.show()

tolerancia=int(0.15*fs)
TP=0
FP=0
FN=0

true_matched=np.zeros_like(peaks_true, dtype=bool) #qué latidos reales ya fueron emparejados

for p in peaks: #comparar cada pico detectado con los verdaderos
    match=False
    for i, pt in enumerate(peaks_true):
        if not true_matched[i] and abs(p-pt)<=tolerancia:
            TP+=1 
            true_matched[i]=True
            match=True
            break
        if not match:
            FP+=1
            
FN=np.sum(~true_matched) #no emparejado -> FN

if TP+FN>0:
    sensibilidad=TP/(TP+FN)
else:
    sensibilidad=0.0
    
if TP+FP>0:
    precision=TP/(TP+FP)
else:
    precision=0.0
    
#Resultados
print(f"latidos reales detectados correctamente (TP): {TP}")
print(f"Falsos positivos (FP): {FP}")
print(f"Falsos negativos (FN): {FN}")
print(f"Sensibilidad: {sensibilidad:.2%}")
print(f"Precisión: {precision:.2%}")
