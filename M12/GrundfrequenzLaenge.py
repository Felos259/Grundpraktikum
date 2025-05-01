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

RF = pd.read_csv('M12/GrundfrequenzLaenge.csv', header=2, sep=';')

dM = 0.000005
Massen = unp.uarray([0.5, 0.2, 0.05], [dM, dM, dM])
uM = Massen[0] + 2*Massen[1] + Massen[2] + 0.05

# Frequenz der Saite = 2 * Anregefrequenz, Unsicherheiten berechnen
for j in range(0, len(RF['df']) , 1):
    RF.loc[j, 'Grundfrequenz'] = RF['Grundfrequenz'][j]*2
    if(RF['Grundfrequenz'][j]<=100):
        RF.loc[j,'df'] = np.sqrt((0.0001 * RF['Grundfrequenz'][j] + 0.02)**2 + 1**2)
    else:
        RF.loc[j,'df'] = np.sqrt((0.0001 * RF['Grundfrequenz'][j] + 0.02)**2 + 1**2) 


uL = unp.uarray(RF['Laenge'], RF['dL'])
uf = unp.uarray(RF['Grundfrequenz'], RF['df'])

# 1 durch f bestimmen
durchf = 1 / uf

# Wert und Unsichheit in Dataset einlesen
RF['durchf']=np.array([value.nominal_value for value in durchf])
RF['deltadurchf']=np.array([value.s for value in durchf])

# Messwerte in CSV schreiben für Einbindung in Latex
header = ["Laenge",'Grundfrequenz', 'df', "durchf", "deltadurchf"]

RF.to_csv('M12/copyReson.csv', sep='&', columns = header, index = False)


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,0.7)
ax.set_ylim(0, 0.008)

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar(RF.Laenge, RF.durchf, xerr=RF.dL , yerr=RF.deltadurchf, label='$\\frac{1}{f}$ in Abhängigkeit der Saitenlänge', color = 'lightblue', linestyle='None', marker='o', capsize=6)

# linearer Fit

# Fitfunktion definieren
def fit_function(x, A):
    return A * x

#Daten
x_data = RF['Laenge']
y_data = RF['durchf']
y_err = RF['deltadurchf'] 

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]

dof = len(RF.index)-len(params)
chi2 = sum([(fit_function(x,A_value)-y)*2/u*2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 10, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Saitenlänge $L$ in m')
plt.ylabel("$\\frac{1}{f}$ in s")
plt.legend()
plt.title("$L$-$\\frac{1}{f}$-Diagramm")

plt.savefig("M12/FrequenzLaenge.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 
#plt.savefig("M12/FrequenzLaenge.svg", format='svg', bbox_inches='tight', pad_inches=0.1) 

# Residuenplot :)
residuen_1 = (y_data - A_value*x_data)*(10**3)

fig, ax = plt.subplots()
ax.set_xlim(0,0.7)
ax.set_ylim(-0.02,0.04)

plt.errorbar(x_data, residuen_1, fmt='o', label='Residuen', capsize=5, color='red')
plt.tick_params(axis="both",direction="in",top=True,left=True,right=True,bottom=True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontale Linie bei y=0
plt.xlabel('Länge $L$ in m')
plt.ylabel('Residuen $(y_i - \hat{y}_i)$ in ms')
plt.legend()
plt.title("Residuendarstellung zum $L$-$\\frac{{1}}{{f}}$-Diagramm")

plt.savefig("ResiduumFrequenzLaenge.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 

plt.show() 

########################################

# keine Ahnung was man beim funktionalen Zusammenhang machen muss

########################################

# Phasengeschwindigkeiten berechnen
uA = u.ufloat(A_value, A_error)

uC =  2.0/uA
print(uC)
