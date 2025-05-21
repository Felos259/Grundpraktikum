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

# Scheinwiderstand bestimmen
uscheinwiderstand = widerstand*uuch1/uuch2 #in kOhm
RF['Z_RL2']=np.array([value.nominal_value for value in uscheinwiderstand])
RF['dZ_RL2']=np.array([value.s for value in uscheinwiderstand])

############################ LINEARER FIT :) #######################################################
# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig2, ax2 = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax2.set_xlim(0,280000)
ax2.set_ylim(0, 10**9)

#Daten
x_data2 = RF['Frequenz']**2
x_err2 = 2*RF['Frequenz']*RF['df'] #Gaußsche Fehlerfortpflanzung
y_data2 = RF['Z_RL2']**2
y_err2 = 2*RF['Z_RL2']*RF['dZ_RL2'] #Gaußsche Fehlerfortpflanzung

# Plot der Messwerte V und p mit Errorbars 
ax2.errorbar(x_data2, y_data2, xerr=x_err2 , yerr=y_err2, label='$|Z_{RL}|^2$ in Abhängigkeit des Quadrats der Frequenz', color = 'lightblue', linestyle='None', marker='o', capsize=6)


# Fitfunktion definieren
def fit_function2(x, A):
    return (161.18)**2+((A * 2*np.pi)**2)*x #A entspricht der Induktivität der Spule, x0 entspricht R_RL

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
induktivitaet2 = unp.uarray([A_value2],[A_error2])
kapazitaet2 = unp.uarray([107.4],[0.2+0.02*107.4])*10**(-6)
print("Die erwartete Resonanzfrequenz liegt bei ",1/unp.sqrt(kapazitaet2*induktivitaet2),"Hz")
print("Die Induktivität der Spule beträgt",induktivitaet2,"H")

x_ax2=np.linspace(0, 600**2, 100000) 
y_ax2 = fit_function2(x_ax2, A_value2)

# Plot zeichnen
plt.plot(x_ax2, y_ax2, label=f"Fit: $y^2 = (2\\cdot \\pi \\cdot A)^2 \\cdot x+x_0^2$ \n $A = {A_value2:.6f} \\pm {A_error2:.6f}$ \n $x_0= {widerstand}$", linewidth=2, color='blue')
plt.xlabel('Frequenz $f^2$ in $s^{-2}$')
plt.ylabel("$|Z_{LR}|^2$ in $(\\Omega)^2$")
plt.legend()
plt.title("$f^2$-$|Z_{LR}|^2$-Diagramm")

plt.savefig("VersuchsteilBLinear.pdf",format='pdf',bbox_inches='tight',pad_inches=0.5)

plt.show()
