import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
import iminuit as i
import uncertainties as u
import uncertainties.umath as um
from uncertainties import unumpy as unp

RF = pd.read_csv('RohdatenAufgabe2.csv', header=1)
plt.rcParams['figure.figsize'] = [19.2,10.8]

T_Luft=[u.ufloat(0,0)]*6
T_Argon=[u.ufloat(0,0)]*6
uKappa_Luft=[u.ufloat(0,0)]*6
uKappa_Argon=[u.ufloat(0,0)]*6
wKappa_Luft=[0]*6
wKappa_Argon=[0]*6

#Werte und Unsicherheiten :)
wVolumen = [4381/(100**3),4325/(100**3)] #in m^3
uVolumen = unp.uarray([4381/(100**3),10/(100**3)],[4325/(100**3),10/(100**3)]) #in m^3
wMasseSchwing = [6.122/1000,6.159/1000] #in kg
uMasseSchwing = unp.uarray([6.122/1000,0.005/1000],[6.159/1000,0.005/1000]) #in kg
wRadius = [13.93/2000,13.95/2000] #in m
uRadius = unp.uarray([13.93/2000,0.01/2000],[13.95/2000,0.01/2000]) #in m
wDruck = 997.046 #hPa, von Frowine
uDruck = u.ufloat(997.046,0.0) #Unsicherheit????

## To-Do: Schwingende Gasmasse ausdenken
wMasse = wMasseSchwing
uMasse = uMasseSchwing

for i in range (0,6,1):
    RF['dT_sys_Luft'+str(i+1)]=0.0005*(RF['Luft'+str(i+1)])+0.03 #sys. Unsicherheit aus Einführungspraktikum
    RF['dT_sys_Argon'+str(i+1)]=0.0005*(RF['Argon'+str(i+1)])+0.03 #sys. Unsicherheit aus Einführungspraktikum

    RF['dT_gross_Luft'+str(i+1)]=0.05 #Größtfehlerabschätzung unserer Langsamkeit in s
    RF['dT_gross_Argon'+str(i+1)]=0.05 #Größtfehlerabschätzung unserer Langsamkeit in s

    RF['dT_Luft'+str(i+1)]=np.sqrt(RF['dT_sys_Luft'+str(i+1)]**2+RF['dT_gross_Luft'+str(i+1)]**2)
    RF['dT_Argon'+str(i+1)]=np.sqrt(RF['dT_sys_Argon'+str(i+1)]**2+RF['dT_gross_Argon'+str(i+1)]**2)
    
    T_Luft[i]=unp.uarray(RF['Luft'+str(i+1)],RF['dT_Luft'+str(i+1)])/100
    T_Argon[i]=unp.uarray(RF['Argon'+str(i+1)],RF['dT_Argon'+str(i+1)])/100

    uKappa_Luft[i]=(4*uVolumen[0]*uMasse[0])/((uRadius[0]**4)*uDruck*(T_Luft[i])**2)
    uKappa_Argon[i]=(4*uVolumen[1]*uMasse[1])/((uRadius[1]**4)*uDruck*(T_Argon[i])**2)

    wKappa_Luft[i]=(4*wVolumen[0]*wMasse[1])/((wRadius[0]**4)*wDruck*(RF['Luft'+str(i+1)]/100)**2)
    wKappa_Argon[i]=(4*wVolumen[0]*wMasse[1])/((wRadius[1]**4)*wDruck*(RF['Argon'+str(i+1)]/100)**2)


print(wKappa_Luft)
meanLuft = np.mean(wKappa_Luft)
standabLuft = np.std(wKappa_Luft,ddof=1)
dmeanLuft = u.ufloat(meanLuft,standabLuft)
print("Ungewichteter Mittelwert Kappa Luft:", dmeanLuft)

meanArgon = np.mean(wKappa_Argon)
standabArgon = np.std(wKappa_Argon,ddof=1)
dmeanArgon = u.ufloat(meanLuft,standabArgon)
print("Ungewichteter Mittelwert Kappa Argon:", dmeanArgon)
