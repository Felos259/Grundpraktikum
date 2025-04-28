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

#WICHTIGE ÄNDERUNGEN ZU MACHEN: Unsicherheiten fixen

RF = pd.read_csv('M12/GrundfrequenzSpannung.csv', header=3, sep=';')

uL = u.ufloat(0.6, 0.006)  #UNSICHERHEIT FIXEN
uM = unp.uarray(RF.loc('Masse'), RF.loc('dM'))

# Massendichte mü berechnen
umu = uM / uL 

# Zugspannungskraft F_0 berechnen - F_0 = i*M*g + F_LH, i = Kerbe

F_LH = 0.52 # in Anleitung gegeben
g = u.ufloat(9.812669,0) #https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html

F_0 = RF.loc('Kerbe') * uM *g + F_LH

