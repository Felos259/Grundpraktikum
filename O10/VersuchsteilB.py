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



#Mittelwert plotten

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.0, 0.1)

# Index renamen damit er bei 1 anfängt
RF.index = np.arange(1, len(RF) + 1)

#Daten
x_data = RF.index
y_data = np.array([value.nominal_value for value in f])
y_err = np.array([value.s for value in f])

#Messwerte plotten
ax.errorbar(x_data, y_data, yerr=y_err, label= 'Brennweite $f_i$' , color = 'mediumblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2 )
# Horizontale Linie bei y=Mittelwert h
plt.axhline(mean - delta_std, color='cornflowerblue', linewidth=1, linestyle='--') 
plt.axhline(mean + delta_std, color='cornflowerblue', linewidth=1, linestyle='--') 
plt.axhline(mean , label = 'Mittelwert $\\Delta \\Overline{f}$ der Brennweiten'  ,color= 'cornflowerblue', linewidth=1, linestyle='-')  


plt.xlabel('i',fontsize=fnt)
plt.ylabel('Brennweite $f$ in EINHEIT!!!!!', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Brennweiten", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O10/Brennweite.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()

