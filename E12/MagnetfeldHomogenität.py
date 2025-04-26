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


mu_0 = 4*np.pi*10**(-7)
N = 320
R = 0.068

k = u.ufloat(mu_0 *N/(2*R)*(4/5)**(3/2),0)

RF = pd.read_csv('E12/BFeldHomogen.csv', header=0, sep=';')

uB = unp.uarray(RF['B'], RF['dB'])
uI = unp.uarray(RF['I'], RF['dI'])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
#Daten
x_data = RF['I']
x_err = RF['dI']
y_data = RF['B']
y_err = RF['dB'] 

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,0.6999)
ax.set_ylim(0, 0.999)

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar( x_data, y_data, xerr=x_err, yerr=y_err, label='Magnetfeldstärke $B$ in Abhängigkeit der Stromstärke $I$', color = 'lightblue', linestyle='None', marker='o', capsize=6)

# linearer Fit
# Fitfunktion definieren
def fit_function(x, A):
    return A * x


# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]

dof = len(RF.index)-len(params)
chi2 = sum([(fit_function(x,A_value)-y)*2/u*2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax = np.linspace(0, 10, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Stromstärke $I$ in A')
plt.ylabel('Magnetfeldstärke $B$ in mT)')
plt.legend() #Legende printen
plt.title("Magnetfeldstärke $B$ in Abhängigkeit der Stromstärke $I$")

plt.savefig("E12/B-I-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.savefig("E12/B-I-Diagramm.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

plt.show() 