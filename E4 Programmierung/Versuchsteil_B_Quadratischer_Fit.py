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
import csv #Output meine Berechnungen in eine CSV-Datei


RF = pd.read_csv('Versuchsteil_B.csv', header=1)

uf = unp.uarray(RF['Frequenz'], RF['df'])
uuch1 = unp.uarray(RF['U_Ch1'], RF['dU_Ch1'])
uuch2 = unp.uarray(RF['U_Ch2'], RF['dU_Ch2'])

# R_RL = Summe aus Omhschen Widerstand und Spulenwiderstand
widerstand = unp.uarray([10.16],[0.2016])+unp.uarray([151.02],[10*0.01+1.5102])
wid = 161.18/1000
uwid=np.sqrt(0.2016**2+(10*0.01+1.5102)**2)/1000

# Scheinwiderstand bestimmen
uscheinwiderstand = widerstand*uuch1/uuch2/1000
RF['Z_RL2']=np.array([value.nominal_value for value in uscheinwiderstand])
RF['dZ_RL2']=np.array([value.s for value in uscheinwiderstand])
print(RF['Z_RL2'])

############################ Quadratischer FIT :) #######################################################
# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig2, ax2 = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax2.set_xlim(0,600)
ax2.set_ylim(0, 1000)

#Daten
x_data2 = RF['Frequenz']
x_err2 = RF['df'] 
y_data2 = RF['Z_RL2']**2
y_err2 = 2*RF['Z_RL2']*RF['dZ_RL2'] #Gaußsche Fehlerfortpflanzung

# Plot der Messwerte V und p mit Errorbars 
ax2.errorbar(x_data2, y_data2, xerr=x_err2 , yerr=y_err2, label='$|Z_{RL}|^2$ in Abhängigkeit der Frequenz', color = 'lightblue', linestyle='None', marker='o', capsize=6)


# Fitfunktion definieren
def fit_function2(x, A):
    return (wid)**2+((A * 2*np.pi*x)**2) #A entspricht der Induktivität der Spule, x0 entspricht R_RL

# Curve-Fit mit Unsicherheiten in y
params2, covariance2 = curve_fit(fit_function2, x_data2, y_data2, sigma=y_err2, absolute_sigma=True)
A_value2 = params2[0]
fit_errors2 = np.sqrt(np.diag(covariance2))  # Fehler der Fit-Parameter
A_error2 = fit_errors2[0]
#x0_value2 = params2[1]
#x0_error2 = fit_errors2[1]

dof2 = len(RF.index)-len(params2)
chi22 = sum([(fit_function2(x,A_value2)-y)**2/u**2 for x,y,u in zip(x_data2,y_data2,y_err2)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value2:.6f} ± {A_error2:.6f}")
#print(f"x0 = {x0_value2:.6f} ± {x0_error2:.6f}")
print(f"Chi-Quadrat/dof: {chi22/dof2}")

#Werteberechnung für Bericht
induktivitaet2 = unp.uarray([A_value2],[A_error2])*1000
kapazitaet2 = unp.uarray([107.4],[0.2+0.02*107.4])*10**(-6)
print("Die erwartete Resonanzfrequenz liegt bei ",1/(2*np.pi*unp.sqrt(kapazitaet2*induktivitaet2)),"Hz")
print("Die Induktivität der Spule beträgt",induktivitaet2,"H")

x_ax2=np.linspace(0, 600, 1000) 
y_ax2 = fit_function2(x_ax2, A_value2)

# Plot zeichnen
plt.plot(x_ax2, y_ax2, label=f"Fit: $y^2 = (2\\cdot \\pi \\cdot A\\cdot x)^2 +x_0^2$ \n $A = {A_value2:.6f} \\pm {A_error2:.6f}$ \n $x_0={wid:.6f} \\pm {uwid:.6f}$", linewidth=2, color='blue')
plt.xlabel('Frequenz $f$ in $s^{-1}$')
plt.ylabel("$|Z_{LR}|^2$ in $(k\\Omega)^2$")
plt.legend()
plt.title("$f$-$|Z_{LR}|^2$-Diagramm")

# Inset
# Inset-Diagramm erstellen
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Position des Inset-Diagramms definieren
ax2_inset = inset_axes(ax2, width="50%", height="50%", loc=2, bbox_to_anchor=(690,-20,300,300)) #Positionierung mit bbox(x-Achse,y-Achse,Größe,Größe)

# Bereich für das Inset-Diagramm
x_inset = np.linspace(-0.18, 140, 1000)
y_inset = fit_function2(x_inset, A_value2)

# Plotten im Inset-Diagramm
ax2_inset.errorbar(RF.Frequenz, RF.Z_RL2, xerr=RF.df , yerr=RF.dZ_RL2, label='$|Z_{RL}|$ in Abhängigkeit der Frequenz', color = 'lightblue', linestyle='None', marker='o', capsize=6)
ax2_inset.plot(x_inset, y_inset, linewidth=2)

# Inset-Bereich anpassen
ax2_inset.set_xlim(-0.18, 125)
ax2_inset.set_ylim(-18, 100)
ax2_inset.set_xticks([0, 100])
ax2_inset.set_yticks([0, 100])
ax2_inset.tick_params(labelsize=8)




plt.savefig("VersuchsteilBQuadratisch.pdf",format='pdf',bbox_inches='tight',pad_inches=0.5)

plt.show()
