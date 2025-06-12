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

# Dinge in Einheiten?
RF = pd.read_csv('O10/Bessel.csv', header=1, sep=';')

P_G = unp.uarray( RF["P_G"], RF["dP_G"] )

P_B = unp.uarray( RF["P_B"], RF["dP_B"] )

P_L = unp.uarray( RF["P_L"], RF["dP_L"] )

P_R = unp.uarray( RF["P_R"], RF["dP_R"] )

l = P_B - P_G

e = P_L - P_R

f = (l**2 - e**2) / (4 * l)

numeric_f = [value.n for value in f]

mean = np.mean(numeric_f)
print('MEAN: ', mean)

std = np.std(numeric_f, ddof=1)
print('STD: ', std)

delta_std = std/np.sqrt(len(numeric_f))
print('DELTA STD: ', delta_std)


