# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:09:26 2023

@author: Simple
"""

##Este código es para extraer los picos y plotear su corrimiento##

##Estaremos usando como referencia el código encontrado en: https://qceha.net/batch_fitting.html ##

##Mientras averiguo cómo extraer los datos directamente desde Peak-o-mat los saqué a Excel manualmente. Esto no es aceptable##

import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

direccion = r'C:\Users\Simple\Documents\U\LAMMPS\Simul temp normal\Temp fix 0nm_1nm 100K_200K zigzag\Peaks_Error_0nm.csv' 
with open(direccion, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    pico1 = [[],[]]
    pico2 = [[],[]]
    pico3 = [[],[]]
    pico4 = [[],[]]
    
    counter=0
    
    for row in csv_reader:
        
        if (counter%2 == 0):
            pico1[0].append(float(row[0]))
            pico2[0].append(float(row[1]))
            pico3[0].append(float(row[2]))
            #pico4[0].append(float(row[3]))
            
        else:
            pico1[1].append(float(row[0]))
            pico2[1].append(float(row[1]))
            pico3[1].append(float(row[2]))
            #pico4[1].append(float(row[3]))
        
        counter+=1
        
temps=np.linspace(100,300,21)

model= LinearRegression()
model.fit(temps.reshape(-1,1),np.array(pico2[0]))

#print(model.coef_[0])
#print(model.intercept_)


plt.figure(figsize=(15,10))
ax = plt.gca()
ax.scatter(temps,pico2[0],label='{:.4f}cm^-1K^-1'.format(model.coef_[0]))
ax.errorbar(temps,pico2[0], pico2[1])
ax.grid()
plt.ylabel('cm$^{-1}$')
plt.xlabel('Temperatura K')
plt.title('Pico ~1540$cm^{-1}$, Cinta 0.69nm x 200nm')
plt.legend()

delta=pico2[0][-1]-pico2[0][0]
kai=model.coef_[0]
freq_0=model.intercept_
eta=1
grun=-1.99

def coef_expansion(delta_freq,kai,freq_0,eta,grun):
    coef=-kai/(eta*grun*(freq_0+delta_freq))
    return coef

print(coef_expansion(delta, kai, freq_0, eta, grun))