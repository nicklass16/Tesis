# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:00:24 2023

@author: Simple
"""

import os
import numpy as np
from tqdm import trange
from scipy.fft import fft
from numba import njit
from datetime import datetime
import time

def corre_sim(ancho, largo, direccion, temperatura, ts):
    
    print('Paso 0: Corre simulacion')
    print('cd ' + direccion + ' && lmp -in in_{}nm_{}nm_{}K_{}TS.airebo'.format(ancho,largo,temperatura,ts))
    os.system('cd ' + direccion + ' && lmp -in in_{}nm_{}nm_{}K_{}TS.airebo'.format(ancho,largo,temperatura,ts))
    print('Simulacion terminada')
    
    vel=open(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),'r') #Archivo de velocidades de LAMMPS
    info=vel.readlines() #Lineas del archivo
    n_atoms=int(info[3]) #Numero de atomos provisto por el archivo
    timez=10000 #Este valor lo ponemos nosotros en la tarjeta de entrada para LAMMPS
    timestep=ts #Cada timestep se toma una medida de las velocidades de las particulas
    timesteps=int(timez/timestep) #Numero de instantes de tiempo en el que medimos las velocidades

    vel_coordenadas=np.ndarray(shape=(timesteps,n_atoms,3)) #El array donde vienen todas las velocidades
    
    print('Paso 1: Guarda las velocidades')

    times=np.zeros(timesteps)
    for t in trange(timesteps):
        times[t]=t*timestep
        for n in range(n_atoms):
            v_atom=info[9+n+(9+n_atoms)*t].split()
            for c in range(3):
                vel_coordenadas[t][n][c]=v_atom[c]
                
    return timesteps, n_atoms, vel_coordenadas, times

def retrieve_sim(ancho, largo, direccion, temperatura, ts):
    
    vel=open(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),'r') #Archivo de velocidades de LAMMPS
    info=vel.readlines() #Lineas del archivo
    n_atoms=int(info[3]) #Numero de atomos provisto por el archivo
    timez=10000 #Este valor lo ponemos nosotros en la tarjeta de entrada para LAMMPS
    timestep=ts #Cada timestep se toma una medida de las velocidades de las particulas
    timesteps=int(timez/timestep) #Numero de instantes de tiempo en el que medimos las velocidades

    vel_coordenadas=np.ndarray(shape=(timesteps,n_atoms,3)) #El array donde vienen todas las velocidades
    
    print('Paso 1: Guarda las velocidades')

    times=np.zeros(timesteps)
    for t in trange(timesteps):
        times[t]=t*timestep
        for n in range(n_atoms):
            v_atom=info[9+n+(9+n_atoms)*t].split()
            for c in range(3):
                vel_coordenadas[t][n][c]=v_atom[c]
                
    return timesteps, n_atoms, vel_coordenadas, times

@njit      
def VACF(direccion, temperatura, ts, timesteps, n_atoms, vel_coordenadas, times):
    
    print('Paso 2: Calcula la funcion de autocorrelacion para todos los tiempos')
    
    #Nota: Siguiendo el codigo de fortran se calcula la funcion de autocorrelacion
    
    Z=np.zeros(timesteps)   #Funcion de autocorrelacion dependiente del tiempo
    
    print('Paso 2.1: Define el denominador de la funcion de autocorrelacion')

    denominador=0.0

    for t0 in range(timesteps):
        for n in range(n_atoms):
            denominador+=np.sqrt(vel_coordenadas[t0][n][0]**2+vel_coordenadas[t0][n][1]**2+vel_coordenadas[t0][n][2]**2)
            
    print('Paso 2.2: Define el numerador de la funcion de autocorrelacion para todos los tiempos')

    numerador=np.zeros(timesteps)

    for t in range(timesteps):
        num=0.0
        for t0 in range(timesteps):
            for n in range(n_atoms):
                if t+t0>timesteps-1:
                    num+=0
                else:
                    num+=np.sqrt(abs(vel_coordenadas[t0+t][n][0]*vel_coordenadas[t0][n][0])+abs(vel_coordenadas[t0+t][n][1]*vel_coordenadas[t0][n][1])+abs(vel_coordenadas[t0+t][n][2]*vel_coordenadas[t0][n][2]))
        numerador[t]=num
        
    Z=numerador/denominador
    
    return Z

def guardar_espectro(direccion, ancho, largo, temperatura, ts, Z, times):
    
    print('Paso 3: Calcula la transformada de Fourier de la funcion de autocorrelacion')
    
    fft_funcion_autocorrelacion=fft(Z)
    
    print('Paso 4: Guarda el espectro')
        
    onda_vs_intensidad=open(direccion+r'\i_vs_k_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),"w")
    
    norm = [float(i)/max(fft_funcion_autocorrelacion) for i in fft_funcion_autocorrelacion]

    for i in range(len(times)):
        onda_vs_intensidad.write(str(times[i]) + " " + str(np.absolute(norm[i])) + "\n")
        
    onda_vs_intensidad.close()
    
def borrar_vel_pos(direccion,ancho,largo,temp,ts):
    
    os.remove(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temp,ts))
    os.remove(direccion+r'\pos_{}nm_{}nm_{}K_{}TS.xyz'.format(ancho,largo,temp,ts))

def espectrografia_LAMMPS(direccion, ancho, largo, temperatura, ts):
    
    dat_sim=retrieve_sim(ancho, largo, direccion, temperatura, ts)
    
    timesteps=dat_sim[0]
    n_atoms=dat_sim[1]
    vel_coordenadas=dat_sim[2]
    times=dat_sim[3]
    
    print('Paso 2: Calcula la funcion de autocorrelacion para todos los tiempos')
    
    Z=VACF(direccion, temperatura, ts, timesteps, n_atoms, vel_coordenadas, times)
    
    guardar_espectro(direccion, ancho, largo, temperatura, ts, Z, times)
    
espectrografia_LAMMPS(r'C:\Users\Simple\Documents\U\LAMMPS\Simul+Datos\Simul Agosto Pruebas\Pruebas viejas\Datos Originales Laura', 5, 25, 400, 20)