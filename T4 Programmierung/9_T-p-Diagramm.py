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

fnt = 25 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

RF = pd.read_csv('T4 Programmierung/Jolly.csv', header=2)

uT = unp.uarray(RF['Temperatur'], RF['dT'])
up =unp.uarray(RF['Druck'], RF['dp'])

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,50)
ax.set_ylim(0, 1200)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.Temperatur, RF.Druck, xerr=RF.dT , yerr=RF.dp, label='$p(T)$', color = '#339999', linestyle='None',  marker='o', markersize=9, capsize=6, elinewidth =2)

# linearer Fit

# Fitfunktion definieren
def fit_function(x, A, x0):
    return A * (x-x0)

#Daten
x_data = RF['Temperatur']
x_err = RF['dT']
y_data = RF['Druck']
y_err = RF['dp'] 

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
x0_value = params[1]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
x0_error = fit_errors[1]

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,A_value,x0_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 100, 1000) 
y_ax = fit_function(x_ax, A_value,x0_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot (x-x_0)$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$ \n $x_0 = {x0_value:.6f} \\pm {x0_error:.6f}$", linewidth=2, color='#a83e9e')

plt.xlabel('Temperatur $T$ in °C', fontsize=fnt)
plt.ylabel("Druck $p$ in hPa", fontsize=fnt)
plt.legend(loc = 'lower right', fontsize=fnt)
plt.title("$T$-$p$-Diagramm", fontsize=fnt)
plt.grid()
ax.set_facecolor("#f9f9f9")
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)  

plt.savefig("T4 Programmierung/T-p-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("T4 Programmierung/T-p-Diagramm.svg", format='svg', bbox_inches='tight', pad_inches=0.5)

# plt.show()
