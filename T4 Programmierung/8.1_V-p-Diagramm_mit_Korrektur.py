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

RF = pd.read_csv('BoyleMariotte.csv', header=4)
RG = pd.read_csv('Totvolumen.csv', header=0)

uL = unp.uarray(RF['Laenge'], RF['dL'])
up =unp.uarray(RF['Druck'], RF['dp'])
uTot=unp.uarray(RG['Totvolumen'],RG['dTotvolumen'])

# Volumen bestimmen
uv = uL*(np.pi*unp.uarray([2.5],[0.005])**2)-uTot #Volumen (inkl. Unsicherheit) - berechnetes Totvolumen
RF['volumen']=np.array([value.nominal_value for value in uv])
RF['delvolumen']=np.array([value.s for value in uv])

# 1 durch p bestimmen
durchp = 1 / up

# Wert und Unsichheit in Dataset einlesen
RF['durchp']=np.array([value.nominal_value for value in durchp])
RF['deldurchp']=np.array([value.s for value in durchp])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,250)
ax.set_ylim(0, 5)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.volumen, RF.durchp, xerr=RF.delvolumen , yerr=RF.deldurchp, label='$1/p$ in Abhängigkeit des Volumens', color = 'lightblue', linestyle='None', marker='o', capsize=6)

# linearer Fit

# Fitfunktion definieren
def fit_function(x, A):
    return A * x

#Daten
x_data = RF['volumen']
x_err = RF['delvolumen']
y_data = RF['durchp']
y_err = RF['deldurchp'] 

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,A_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 250, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Volumen $V$ in $cm^3$')
plt.ylabel("$1/p$ in $\\text{bar}^{-1}$")
plt.legend()
plt.title("$V$-$\\frac{1}{p}$-Diagramm")

plt.savefig("1durchpVDiagramm_korr.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.savefig("1durchpVDiagramm_korr.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

# plt.show()


