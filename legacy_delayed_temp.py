# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:17:48 2023

@author: Simple
"""

import os
import numpy as np
from tqdm import trange
from scipy.fft import fft
from numba import njit

def corre_sim(direccion, ancho, largo, temperatura, ts):
    
    print('Paso 0: Corre simulacion')
    os.system('cd ' + direccion + ' && lmp -in in_{}nm_{}nm_{}K_{}TS.airebo'.format(ancho,largo,temperatura,ts))
    print('Simulacion terminada')
    
    vel=open(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),'r') #Archivo de velocidades de LAMMPS
    info=vel.readlines() #Lineas del archivo
    n_atoms=int(info[3]) #Numero de atomos provisto por el archivo
    time=10000 #Este valor lo ponemos nosotros en la tarjeta de entrada para LAMMPS
    timestep=ts #Cada timestep se toma una medida de las velocidades de las particulas
    timesteps=int(time/timestep) #Numero de instantes de tiempo en el que medimos las velocidades
    k_b=0.8314472456714728 #Constante de Boltzmann calculada
    n_dim=3 #Dimensiones en las que se corre la simulación
    N_dof=n_dim*n_atoms-n_dim
    m=12.02 #Masa de los átomos de carbono
    
    vel_coordenadas=np.ndarray(shape=(timesteps,n_atoms,3)) #El array donde vienen todas las velocidades
    e_kin=np.ndarray(shape=(timesteps)) #El array de energías para cada timestep
    temp_cal=np.ndarray(shape=(timesteps)) #El array de temperaturas para cada timestep
    
    print('Paso 1: Guarda las velocidades')

    times=np.zeros(timesteps)
    for t in trange(timesteps):
        times[t]=t*timestep
        for n in range(n_atoms):
            v_atom=info[9+n+(9+n_atoms)*t+(9+n_atoms)*timesteps].split()
            for c in range(3):
                vel_coordenadas[t][n][c]=v_atom[c]
            e_kin[t]+=0.5*m*(vel_coordenadas[t][n][0]**2+vel_coordenadas[t][n][1]**2+vel_coordenadas[t][n][2]**2)

    for t in range(timesteps):
        temp_cal[t]=(2*e_kin[t])/(N_dof*k_b)
                
    return timesteps, n_atoms, vel_coordenadas, times, temp_cal

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

def guardar_espectro(direccion, ancho, largo, temperatura, ts, Z, times, temp_cal):
    
    print('Paso 3: Calcula la transformada de Fourier de la funcion de autocorrelacion')
    
    fft_funcion_autocorrelacion=fft(Z)
    
    print('Paso 4: Guarda el espectro')
        
    onda_vs_intensidad=open(direccion+r'\i_vs_k_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),"w")
    temp_calculada=open(direccion+r'\temp_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),'w')
    
    norm = [float(i)/max(fft_funcion_autocorrelacion) for i in fft_funcion_autocorrelacion]

    for i in range(len(times)):
        onda_vs_intensidad.write(str(times[i]) + " " + str(norm[i].real) + "\n")
        temp_calculada.write(str(times[i]) + " " + str(temp_cal[i]) + "\n")
        
    onda_vs_intensidad.close()
    temp_calculada.close()

def espectrografia_LAMMPS(direccion, ancho, largo, temperatura, ts):
    
    dat_sim=corre_sim(direccion, ancho, largo, temperatura, ts)
    
    timesteps=dat_sim[0]
    n_atoms=dat_sim[1]
    vel_coordenadas=dat_sim[2]
    times=dat_sim[3]
    temp_cal=dat_sim[4]
    
    print('Paso 2: Calcula la funcion de autocorrelacion para todos los tiempos')
    Z=VACF(direccion, temperatura, ts, timesteps, n_atoms, vel_coordenadas, times)
    
    guardar_espectro(direccion, ancho, largo, temperatura, ts, Z, times, temp_cal)
    
    borrar_vel_pos(direccion, ancho, largo, temperatura, ts)
    
def borrar_vel_pos(direccion,ancho,largo,temp,ts):
    
    os.remove(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temp,ts))
    os.remove(direccion+r'\pos_{}nm_{}nm_{}K_{}TS.xyz'.format(ancho,largo,temp,ts))
    
def create_new_in_LAMMPS(direccion, ancho, largo, temperatura, ts):
    
    fin = open(direccion+r'\in_0nm_200nm_100K_5TS.airebo','r')
    
    content=fin.readlines()
    
    fin.close()
    
    read_5 = 'read_data           {}nm_{}nm.airebo'.format(ancho,largo)    
    vel_15= 'velocity            all create {}.0 761341'.format(temperatura)
    fix_17= 'fix                 1 all nvt temp {} {} $(100.0*dt)'.format(temperatura,temperatura)
    dump1_21= 'dump 	        4 all custom 5 vel_{}nm_{}nm_{}K_{}TS.dat vx vy vz'.format(ancho,largo,temperatura,ts)
    dump2_22= 'dump            3 all xyz 10 pos_{}nm_{}nm_{}K_{}TS.xyz'.format(ancho,largo,temperatura,ts)
    
    new_in=open(direccion+r'\in_{}nm_{}nm_{}K_{}TS.airebo'.format(ancho,largo,temperatura,ts),'w')
    
    for i in range(len(content)):
        if i==5:
            new_in.write(read_5+'\n')
        elif i==15:
            new_in.write(vel_15+'\n')
        elif i==17:
            new_in.write(fix_17+'\n')
        elif i==21:
            new_in.write(dump1_21+'\n')
        elif i==22:
            new_in.write(dump2_22+'\n')
        else:
            new_in.write(content[i])
            
    new_in.close()

anchos=[0,1]
temps=np.linspace(100,450,36)

for i in range(len(anchos)):
    for j in range(len(temps)):
        espectrografia_LAMMPS(r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Temp fix delay armchair', int(anchos[i]), 200, int(temps[j]), 5)