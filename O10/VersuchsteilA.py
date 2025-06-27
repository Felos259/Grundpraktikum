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

# Radius des Sphärometers in mm
r = 15

deltah = np.sqrt(0.007**2+0.005**2)
print("Delta h: " + str(deltah) + 'mm')

# Gemessene Höhen in mm mit Unsicherheit
h_1 = u.ufloat(0.635, deltah) 
h_2 = u.ufloat(0.635, deltah) 

R_1 = r**2/(2*h_1) + h_1/2
R_2 = r**2/(2*h_2) + h_2/2


print('Krümmungsradius R_1: ' + str(R_1) + 'mm')

print('Krümmungsradius R_2: ' + str(R_2) + 'mm')
# Brechungszahl des Materials
n = 1.52

# Brennweite bestimmen
f = 1/((n-1) * ((1/R_1) + (1/R_2)))

print('Brennweite: ' + str(f) + 'mm')