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

fnt = 15 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

RF = pd.read_csv('T4 Programmierung/BoyleMariotte.csv', header=3)

uL = unp.uarray(RF['Laenge'], RF['dL'])
up =unp.uarray(RF['Druck'], RF['dp'])
print(RF['dp'])

# Volumen bestimmen
uv = uL*(np.pi*unp.uarray([2.5],[0.005])**2)-20 #Volumen (inkl. Unsicherheit) - Totvolumen aus Anleitung
RF['volumen']=np.array([value.nominal_value for value in uv])
RF['delvolumen']=np.array([value.s for value in uv])

# 1 durch p bestimmen
durchp = 1 / up

# Wert und Unsichheit in Dataset einlesen
RF['durchp']=np.array([value.nominal_value for value in durchp])
RF['deldurchp']=np.array([value.s for value in durchp])

print(RF['durchp'],RF['deldurchp'])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,375)
ax.set_ylim(0.000001, 1.5)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.volumen, RF.durchp, xerr=RF.delvolumen , yerr=RF.deldurchp, label='$\\frac{1}{p(V)}$', color = '#b2dcb6', linestyle='None', elinewidth =1.5, marker='o', capsize=6)

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

x_ax=np.linspace(0, 500, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='#a83e9e')

plt.xlabel('Volumen $V$ (inklusive $20$ $cm^3$ Totvolumen) in $cm^3$', fontsize=fnt)
plt.ylabel("$\\frac{1}{p}$ in $\\text{bar}^{-1}$", fontsize=fnt)
plt.legend(loc = 'lower right', fontsize=fnt)
plt.grid()
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)  

plt.title("$V$-$\\frac{1}{p}$-Diagramm", fontsize=fnt)
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)  


plt.savefig("T4 Programmierung/1durchpVDiagramm_alt.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("T4 Programmierung/1durchpVDiagramm_alt.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 


plt.show()



