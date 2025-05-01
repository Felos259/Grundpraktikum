import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
import uncertainties as u
import uncertainties.umath as um
from uncertainties import unumpy as unp


RF = pd.read_csv('M12/FehlerRaus/mu.csv', header=0, sep=';')

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,4.1)
ax.set_ylim(0.0008,0.00089)

x_data = [1,2,3,4]
y_data = RF['mu']
y_err = RF['dMu'] 

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar(x_data, y_data, yerr=y_err, label='Messwerte für $\\mu$', 
            color = 'lightblue', linestyle='None', marker='o', capsize=6)

plt.xlabel('n')
plt.ylabel('$\\mu_n$ in $\\frac{kg}{m}$')
plt.legend()
plt.title("$\\mu$-Diagramm")

plt.savefig("M12/FehlerRaus/MuPlot.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 
plt.show()