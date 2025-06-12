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

r = u.ufloat(0.0, 0.0)

# Gemessene Höhen in EINHEIT
h_1 = u.ufloat(0.0, 0.0) 
h_2 = u.ufloat(0.0, 0.0) 

R_1 = r**2/(2*h_1) + h_1/2
R_2 = r**2/(2*h_2) + h_2/2

print('Krümmungsradius R_1: ' + R_1 )

print('Krümmungsradius R_2: ' + R_2 )

# Brechungszahl des Materials
n = 0

# Brennweite bestimmen
f = 1/((n-1) * ((1/R_1) + (1/R_2)))

print('Brennweite: ' + f)