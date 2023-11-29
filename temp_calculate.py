# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:57:29 2023

@author: Simple
"""

##Temperature calculations from velocities##

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

direccion=r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Temp fix dump TEST'
ancho=0
largo=200
temperatura=100
ts=5
vel=open(direccion+r'\vel_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temperatura,ts),'r') #Archivo de velocidades de LAMMPS
info=vel.readlines() #Lineas del archivo
n_atoms=int(info[3]) #Numero de atomos provisto por el archivo
time=10000 #Este valor lo ponemos nosotros en la tarjeta de entrada para LAMMPS
timestep=ts #Cada timestep se toma una medida de las velocidades de las particulas
timesteps=int(time/timestep) #Numero de instantes de tiempo en el que medimos las velocidades
k_b=0.8314472456714728 #Constante de Boltzmann calculada
n_dim=3
N_dof=n_dim*n_atoms-n_dim

vel_coordenadas=np.ndarray(shape=(timesteps,n_atoms,3)) #El array donde vienen todas las velocidades
e_kin=np.ndarray(shape=(timesteps)) #El array de energías para cada timestep
temp=np.ndarray(shape=(timesteps)) #El array de temperaturas para cada timestep

print('Paso 1: Guarda las velocidades')

m=12.02

times=np.zeros(timesteps)
for t in trange(timesteps):
    times[t]=t*timestep
    for n in range(n_atoms):
        v_atom=info[9+n+(9+n_atoms)*t].split()
        for c in range(3):
            vel_coordenadas[t][n][c]=v_atom[c]
        e_kin[t]+=0.5*m*(vel_coordenadas[t][n][0]**2+vel_coordenadas[t][n][1]**2+vel_coordenadas[t][n][2]**2)

for t in range(timesteps):
    temp[t]=(2*e_kin[t])/(N_dof*k_b)