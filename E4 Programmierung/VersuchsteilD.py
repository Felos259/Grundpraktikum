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

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

RF = pd.read_csv('E4 Programmierung\Versuchsteil_D.csv', header=1, sep=',')

mean = np.mean(RF['Frequenz'])

# korrigierte Standardabweichung berechnen
std = np.std(RF['Frequenz'], ddof=1)

deltaStd = std * np.sqrt(10)

deltaFBar = np.sqrt(deltaStd**2+0.05**2)

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0.5,10.5)
ax.set_ylim(688, 693)

# Index renamen damit er bei 1 anfängt
RF.index = np.arange(1, len(RF) + 1)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.index, RF.Frequenz , yerr=RF.dF , label='gemessene Resonanzfrequenzen', color = 'lightblue', linestyle='None', marker='o', capsize=6)
plt.axhline(mean, color='blue', linewidth=0.8, linestyle='-', label = 'Mittelwert mit Unsicherheit') 
plt.axhline(mean - deltaFBar, color='blue', linewidth=0.8, linestyle='--') 
plt.axhline(mean + deltaFBar, color='blue', linewidth=0.8, linestyle='--') 


plt.xlabel('n', fontsize=fnt)
plt.ylabel('Frequenz $f$ in Hz', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Resonanzfrequenzen des RCL-Schwingkreises", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("E4 Programmierung/Resonanzfrequenzen.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()
