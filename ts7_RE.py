# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 08:39:15 2025

@author: Catalina Re
"""
#Usando el archivo ecg.mat que contiene un registro electrocardiográfico (ECG) tegistrado durante una prueba de esfuerzo, junto con una serie de vriables descriptas a continuación.
#Disee y aplique los filtros digitales necesarios para mitigar las siguientes fuentes de contaminación:
#   - Ruido causado por le movimiento de los electrodos (Alta frecuencia)
#   - Ruido muscular (Alta frecuencia)
#   - Movimiento de la línea de base del ECG, inducido en parte por la respiración (Baja frecuencia)

#Variables del archivo ecg.mat
#   - ecg_lead: registro de ECG muestreado a fs=1 fs=1 kHz durante una prueba de esfuerzo
#   - qrs_pattern1: complejo de ondas QRS normal
#   - hearbeat_pattern1: latido normal
#   - hearbeat_pattern2: latido de origen ventricular
#   - qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren los latidos

#a) establezca una plantilla de diseño para los filtros digitales que necesitará para que la señal de ECG se asemeje a los latidos promedio en cuanto a suavidad a trazos y nivel isoeléctrico nulo
#   (utilice los resultados del ancho de banda estimado del ECG en la TS5. Tome como referencia las siguientes morfologías promedio para evaluar cualitativamente la efectividad de los filtros diseñados)
#b) ¿cómo obtuvo dichos valores? Describa el procedimiento para establecer los valores de la plantilla
#c) diseñe al menos dos filtros FIR y dos IIR para su comparación. Verifique que la respuesta en frecuencia responda a la plantilla de diseño
#   (para los filtros IIR adopte las aproximaciones de módulo de máxima planicidad, Chebyshev y Cauer. Para los filtros FIR, utilice las metodologías de ventanas, cuadrados mínimos y Parks-Mc Clellan-Remez)
#d) Evalúe el rendimiento de los filtros que haya diseñado:
#   - Verifique que filtra las señales interferentes
#   - Verifique que es inocuo en las zonas donde no hay interferencias
#   (utilice el código como referencia para analizar los puntos 1 y 2. También pude incluir otras regiones que considere de interés)

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla #plantilla

def plot_regiones (ecg_signal, ecg_signal_filtr, reg_interes, demora, label='ECG filtrado', crear_figura=True):
    c_muestras = len(ecg_signal) #cantidad de muestras
    for i in reg_interes:
        zoom = np.arange(max(0, i[0]), min(c_muestras, i[1]), dtype='uint')
        if crear_figura:
            plt.figure(figsize=(16,8), facecolor='w', edgecolor='k')
        plt.plot(zoom, ecg_signal[zoom], label='ECG', linewidth=2)
        plt.plot(zoom, ecg_signal_filtr[zoom+demora], label=label)
        plt.title('ECG filro ejemploo desde '+str(i[0])+' a '+str(i[1]))
        plt.ylabel('Adimensional')
        plt.xlabel('Muestras (#)')
        axes_hdl=plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())
        
        if crear_figura:
            plt.show()
        
#extraer y visualizar datos del ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead=mat_struct['ecg_lead'].flatten()
N1=len(ecg_one_lead)

ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead) #normalizar la señal

#definir las variables a analizar
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

senales= {
    'QRS (latido normal)': qrs_pattern1,
    'Latido normal completo': heartbeat_pattern1,
    'Latido ventircular': heartbeat_pattern2,
    'Señal con detecciones QRS': qrs_detections
    }

fig,axs=plt.subplots(2,2,figsize=(12,10), sharey=False)
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

#plantilla de filtro
fs=1000 #Hz
nyq_frec=fs/2 
ripple=1 #dB
attenuation=40 #dB
fpass=np.array([1.0, 35.0]) #Hz
fstop=np.array([0.1, 50.0]) #Hz

plt.figure(figsize=(10,6))
plt.title('Plantilla de diseño filtro pasabanda para ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()

#%% FILTROS IIR
aprox_name=['butter', 'cheby1']
filtros_i={}
fig,axs=plt.subplots(1,2,sharey=True)
axs=axs.flatten()

w_rad=np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad=np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True))/nyq_frec*np.pi

for idx, ftype in enumerate(aprox_name):
    mi_sos=sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=ftype, output='sos', fs=fs)
    filtros_i[ftype.capitalize()]=mi_sos
    w, hh=sig.sosfreqz(mi_sos, worN=w_rad)
    
    axs[idx].plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15),label=ftype)
    axs[idx].set_title(f'Aproximación:{ftype}')
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].grid(True)
    axs[idx].legend()
    
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
    
plt.suptitle('Filtros IIR', fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

#%% FILTROS FIR
freq=[0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq_frec]
gain=[0,0,1,1,0,0]

filtro_ventana=sig.firwin2(numtaps=2505, freq=freq, gain=gain, window='hamming', fs=fs)

numtaps_ls=1505
filtro_ls=sig.firwin2(numtaps_ls, freq, gain, fs=fs)

filtros_fir={
    'Método de ventanas': filtro_ventana,
    'Método de cuadrados mínimos': filtro_ls
    }

fig, axs=plt.subplots(1,2,figsize=(12,5), sharey=True)
axs=axs.flatten()

for idx, (nombre, filtro) in enumerate(filtros_fir.items()):
    w,h=sig.freqz(filtro, worN=8000, fs=fs)
    axs[idx].plot(w, 20*np.log10(np.abs(h)+1e-15), label=nombre)
    axs[idx].set_title(nombre)
    axs[idx].set_xlabel('Frecuencia [Hz')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].set_ylim(-100,10)
    axs[idx].grid(True)
    axs[idx].legend()
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
    
plt.suptitle('Filtros FIR', fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

#diseño por simetrización
freq=[0,fstop[0],fpass[0],fpass[1],fpass[1]+1,nyq_frec]
gain=[0,0,1,1,0,0]

filtro_ventana2=sig.firwin2(numtaps=2505, freq=freq, gain=gain, window='hamming', fs=fs)

numptaps_ls=1505
filtro_ls=sig.firls(numtaps_ls, freq, gain, fs=fs)

filtros_fir={
    'Método de ventanas': filtro_ventana2,
    'Método de cuadrados mínimos': filtro_ls
    }

fig, axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
axs=axs.flatten()

for idx, (nombre, filtro) in enumerate(filtros_fir.items()):
    w,h=sig.freqz(filtro, worN=8000, fs=fs)
    axs[idx].plot(w, 20*np.log10(np.abs(h)+1e-15),label=nombre)
    axs[idx].set_title(nombre)
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].set_ylim(-100,10)
    axs[idx].grid(True)
    axs[idx].legend()
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
    
plt.suptitle('Filtros FIR por simetrización', fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

#diseño por concatenación
def filtro_pasabanda (signal, filtro_alto, filtro_bajo):
    salida1=np.convolve(signal,filtro_alto, mode='same')
    salida2=np.convolve(salida1, filtro_bajo, mode='same')
    return salida2

#   diseño de filtros concatenados - ventanas
filtro_pasaalto=sig.firwin(numtaps=501, cutoff=fpass[0], fs=fs, pass_zero=False, window='hamming')
filtro_pasabajo=sig.firwin(numtaps=501, cutoff=fpass[1], fs=fs, pass_zero=True, window='hamming')

signal_filtrada=filtro_pasabanda(ecg_one_lead, filtro_pasaalto, filtro_pasabajo) #aplicar filtro pasabanda

filtro_pbanda=np.convolve(filtro_pasaalto, filtro_pasabajo) #mostrar respuesta en frec del filtro combinado

w,h=sig.freqz(filtro_pbanda, worN=8000, fs=fs)

plt.figure(figsize=(10,4))
plt.plot(w, 20*np.log10(np.abs(h)+1e-15))
plt.title('Filtro pasa banda por ventanas - concatenación')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.ylim([-100,10])
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()

#   diseño con mínimos cuadrados
#   pasa altos atenúa <0,1Hz, pasa >1Hz
freq_pa=[0, fstop[0], fpass[0], nyq_frec]
gain_pa=[0,0,1,1]
numtaps=1501
filtro_pasaalto=sig.firls(numtaps, freq_pa, gain_pa, fs=fs)

#   pasa bajos: pasa <35Hz, atenúa >50Hz
freq_pb=[0,fpass[1], fstop[1], nyq_frec]
gain_pb=[1,1,0,0]
numtaps=501
filtro_pasabajo=sig.firls(numtaps, freq_pb, gain_pb, fs=fs)

signal_filtrada=filtro_pasabanda(ecg_one_lead, filtro_pasaalto, filtro_pasabajo) #aplicar filtro pasa banda
filtro_pbanda=np.convolve(filtro_pasaalto, filtro_pasabajo) #mostrar rta en frec del filtro combinado

w,h=sig.freqz(filtro_pbanda, worN=8000, fs=fs)

plt.figure(figsize=(10,4))
plt.plot(w, 20*np.log10(np.abs(h)+1e-15))
plt.title('Filtro pasabanda por cuadrados mínimos (concatenación)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]]')
plt.ylim([-100,10])
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()

filtros_fir['Método de cuadrados mínimos']=filtro_pbanda
filtros_fir['Método de ventanas']=filtro_ventana

#%% ANÁLISIS DE FILTRO IIR
ecgs_filtrados={}
nombres_filtros=['Butter', 'Cheby1']

for nombre, filtro in filtros_i.items():
    ecg_filt=sig.sosfilt(filtro,ecg_one_lead)
    ecgs_filtrados[nombre]=ecg_filt
    plt.figure()
    plt.plot(ecg_one_lead, label='Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada')
    plt.title(f'ECG filtrado - {nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
#análisis de regiones de interes
reg_1=(
        np.array([5,5.2])*60*fs,
        np.array([12,12.4])*60*fs,
        np.array([15,15.2])*60*fs,
        )

demora=10 

for nombre, ecg_filt in ecgs_filtrados.items():
    fig,axs=plt.subplots(1,len(reg_1), figsize=(18,5), sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_1):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
    
    plt.suptitle(f'{nombre} - Regiones sin ruido', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
reg_2=(
    [4000,5500],
    [10000,11000]
    )
demora2=67
    
for nombre, ecg_filt in ecgs_filtrados.items():
    fig,axs=plt.subplots(1, len(reg_2), figsize=(12,5), sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_2):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región ruido {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
        
    plt.suptitle(f'{nombre} - Regiones con ruido', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
ecgs_filtrados={}
nombres_filtros=['Butter', 'Cheby1']

for nombre, filtro in filtros_i.items():
    ecg_filt=sig.sosfiltfilt(filtro, ecg_one_lead)
    ecgs_filtrados[nombre]=ecg_filt

reg_1=(
       np.array([5,5.2])*60*fs,
       np.array([12,12.4])*60*fs, 
       np.array([15,15.2])*60*fs
       )

demora=10 

for nombre, ecg_filt in ecgs_filtrados.items():
    fig,axs=plt.subplots(1, len(reg_1), figsize=(18,5), sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_1):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Regiónn {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
    
    plt.suptitle(f'{nombre} - Región sin ruido', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
reg_2=(
       [4000,5500],
       [10000,11000]
       )

demora2=10 

for nombre, ecg_filt in ecgs_filtrados.items():
    fig,axs=plt.subplots(1,len(reg_2),figsize=(12,5),sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_2):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead,ecg_filt, [reg],demora2,label=nombre,crear_figura=False)
        axs[i].set_title(f'{nombre} - Región ruido {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
        
    plt.suptitle(f'{nombre} - Regiones con ruido', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

#%% ANÁLISIS DE FILTRO FIR
ecgs_filtrados_fir={}
nombre_filtros=['Ventanas', 'Cuadrados mínimos']

for nombre, filtro in filtros_fir.items():
    ecg_filt=np.convolve(ecg_one_lead, filtro, mode='same')
    ecgs_filtrados_fir[nombre]=ecg_filt
    plt.figure()
    plt.plot(ecg_one_lead, label='Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada')
    plt.title(f'ECG filtrado - {nombre}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
#análisis de regiones de interés
reg_1=(
       np.array([5,5.2])*60*fs,
       np.array([12,12.4])*60*fs,
       np.array([15,15.2])*60*fs
       )

demora=10 

for nombre, ecg_filt in ecgs_filtrados_fir.items():
    gis,axs=plt.subplots(1,len(reg_1), figsize=(18,5), sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_1):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
    
    plt.suptitle(f'{nombre} Regiones de interés 1', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
reg_2=(
       [4000,5500],
       [10000,11000]
       )

demora2=0

for nombre, ecg_filt in ecgs_filtrados_fir.items():
    fig,axs=plt.subplots(1,len(reg_2),figsize=(12,5),sharey=True)
    axs=axs.flatten()
    
    for i, reg in enumerate(reg_2):
        plt.sca(axs[i])
        plot_regiones(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región 2 {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)
        
    plt.suptitle(f'{nombre} - Regiones de interés 2', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
