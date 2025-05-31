# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:02:40 2025

@author: Catalina Re
"""

# Dadas las siguientes ecuaciones de diferencias de los siguientes sistemas, que representan 
# un filtro de media móvil:
# a. y(n)=x(n-3)+x(n-2)+x(n-1)+x(n)
# b. y(n)=x(n-4)+x(n-3)+x(n-2)+x(n-1)x(n)
# c. y(n)=x(n)-x(n-1)
# d. y(n)=x(n)-x(n-2)
# Se pide:
#   - Hallar T(z)=Y(z)/X(z)
#   - Calcular su respuesta en frecuencia de módulo y fase
#   - Simular y validar la respuesta en frecuencia de todos los sistemas con Numpy

import numpy as np
import matplotlib.pyplot as plt

#Números de puntos de frecuncia
N = 1024
w = np.linspace(0, np.pi, N) #frecuencia angular de 0 a pi

#Funciones de transferencia
T_a = (1+np.exp(1j*w)+np.exp(2j*w)+np.exp(3j*w))/np.exp(3j*w)
T_b = (1+np.exp(1j*w)+np.exp(2j*w)+np.exp(3j*w)+np.exp(4j*w))/np.exp(4j*w)
T_c = 1-np.exp(-1j*w)
T_d = 1-np.exp(-2j*w)

# a. 
fase_a = np.angle(T_a)
modulo_a = np.abs(2*np.cos(w*3/2)+2*np.cos(w*1/2))

# b. 
fase_b = np.angle(T_b)
modulo_b = np.abs(1+(2*np.cos(2*w))+(2*np.cos(w)))
                  
# c.
fase_c = np.angle(T_c)
modulo_c = np.abs(2*np.sin(w/2))

# d. 
fase_d = np.angle(T_d)
modulo_d = np.abs(2*np.sin(w))

#%% GRÁFICOS

plt.figure()

plt.subplot(2,1,1)
plt.plot(w, modulo_a)
plt.title ('Respuesta en frecuencia a)')
plt.ylabel('Módulo')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w, fase_a)
plt.ylabel('Fase [rad]')
plt.grid(True)

plt.figure()

plt.subplot(2,1,1)
plt.plot(w, modulo_b)
plt.title ('Respuesta en frecuencia b)')
plt.ylabel('Módulo')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w, fase_b)
plt.ylabel('Fase [rad]')
plt.grid(True)

plt.figure()

plt.subplot(2,1,1)
plt.plot(w, modulo_c)
plt.title ('Respuesta en frecuencia c)')
plt.ylabel('Módulo')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w, fase_c)
plt.ylabel('Fase [rad]')
plt.grid(True)

plt.figure()

plt.subplot(2,1,1)
plt.plot(w, modulo_d)
plt.title ('Respuesta en frecuencia d)')
plt.ylabel('Módulo')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w, fase_d)
plt.ylabel('Fase [rad]')
plt.grid(True)

#%% EXTRA
# #Respuesta en frecuencia
# T_a = np.exp(-3j*w)+np.exp(-2j*w)+np.exp(-1j*w)+1
# T_b = np.exp(-4j*w)+np.exp(-3j*w)+np.exp(-2j*w)+np.exp(-1j*w)+1
# T_c = 1-np.exp(-1j*w)
# T_d = 1-np.exp(-2j*w)

# #Modulo y fase
# mod_a = np.abs(T_a)
# mod_b = np.abs(T_b)
# mod_c = np.abs(T_c)
# mod_d = np.abs(T_d)

# fase_a = np.angle(T_a)
# fase_b = np.angle(T_b)
# fase_c = np.angle(T_c)
# fase_d = np.angle(T_d)

# plt.figure()

# plt.subplot(2,1,1)
# plt.plot(w, 20*np.log10(mod_a))
# plt.title('Respuesta en frecuencia a)')
# plt.ylabel('Módulo [dB]')
# plt.grid(True)

# plt.subplot(2,1,2)
# plt.plot (w, fase_a)
# plt.xlabel('Frecuencia [rad/muestra]')
# plt.ylabel('Fase [rad]')
# plt.grid(True)
