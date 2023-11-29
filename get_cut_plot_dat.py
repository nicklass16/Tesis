# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:45:35 2023

@author: Simple
"""

import numpy as np
import matplotlib.pyplot as plt

def get_dat(direccion, ancho, largo, temp, ts):
    
    data_file = direccion + r'\i_vs_k_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temp,ts)
        
    tau = np.array([])
    i = np.array([])
    
    with open(data_file, 'r') as file:
        for line in file:
            line_data = line.strip().split()
            if len(line_data) == 2:
                tau = np.append(tau,float(line_data[0]))
                i = np.append(i,float(line_data[1]))
                
    #data_array = np.column_stack((tau,i))
    
    return tau,i

def cut_dat(dat,rango,ts):
    
    tau=dat[0]
    i=dat[1]
    indice_l=int(rango[0]/ts)
    indice_h=int(rango[1]/ts)

    tau_c=tau[indice_l-1:indice_h]
    i_c=i[indice_l-1:indice_h]
    
    return tau_c, i_c

def save_dat(direccion,rango,ancho,largo,temp,ts):
    
    dat=get_dat(direccion, ancho, largo, temp, ts)
    dat_cuted=cut_dat(dat,rango,ts)
    
    
    file=open(direccion+r'\{}nm_{}nm_{}K_{}TS_{}_{}.txt'.format(ancho,largo,temp,ts,rango[0],rango[1]), 'a')
    
    for i in range(len(dat_cuted[0])):
        file.write(str(dat_cuted[0][i]) + ' ' + str(dat_cuted[1][i]) + '\n')
        
    file.close()
    
def get_temp(direccion,ancho,largo,temp,ts):
    
    data_file = direccion + r'\temp_{}nm_{}nm_{}K_{}TS.dat'.format(ancho,largo,temp,ts)
        
    temperatura = np.array([])
    i = np.array([])
    
    with open(data_file, 'r') as file:
        for line in file:
            line_data = line.strip().split()
            if len(line_data) == 2:
                i = np.append(i,float(line_data[0]))
                temperatura = np.append(temperatura,float(line_data[1]))
                
    for t in range(len(temperatura)):
        if temperatura[t]>4*temp:
            #print(temperatura[t])
            temperatura[t]=temp
        elif temperatura[t]<0:
            print(temperatura[t])
            temperatura[t]=temp
    
    return temperatura,i

#largos=np.linspace(150,250,11)
temps=np.linspace(150,250,11)

for i in range(len(temps)):
    save_dat(r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Temp fix 0nm_1nm 100K_200K zigzag', [1200,1600], 0, 200, int(temps[i]), 5)

#dat=get_dat(r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Temp fix 0nm_1nm 100K_200K zigzag',0,200,100,5)
#dat_cuted=cut_dat(dat,[1200,1700],5)

#temps=[100]


plt.figure(figsize=(20,15))
for i in range(len(temps)):
    dat=get_dat(r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Gruneisen Parameter',0,int(temps[i]),180,5)
    dat_cuted=cut_dat(dat,[1200,1600],5)
    ax = plt.gca()
    ax.plot(-dat_cuted[0], -dat_cuted[1],label='{}K'.format(temps[i]))
    ax.grid()
    ax.axes.yaxis.set_ticklabels([])
    plt.legend()
plt.axvline(x = -1331, color = 'black', ls=':', label='Pico ~1331 cm-1')
plt.axvline(x = -1420, color = 'black', ls=':', label='Pico ~1420 cm-1')
plt.axvline(x = -1540, color = 'black', ls=':', label='Pico ~1540 cm-1')
plt.xlabel('1/cm')
plt.ylabel('Intensidad relativa')
plt.title('Cinta 0.69nm x 200nm, espectro 1200_1600')
plt.show()

titi=get_temp(r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Gruneisen Parameter',0,200,180,5)
plt.figure(figsize=(20,15))
plt.plot(titi[1],titi[0])
plt.title('Temperatura GNR3 controlada, @100K')
ax = plt.gca()
ax.grid()
