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
RF = pd.read_csv('O10/Abbe.csv', header=2, sep=';')

d = 1.0

array = [d, d, d, d, d, d, d, d, d, d]

d = 0.3
P_G = unp.uarray( RF["P_G"], array)

d = 0.1
P_K = unp.uarray( RF["P_K"], array)

G = RF["G"]

B = unp.uarray( RF["B"], RF["dB"])

Gstrich = RF["G'"]

Bstrich = unp.uarray( RF["B'"], RF["dB'"])

y = P_K - P_G

gamma = B/G

gammastrich = Bstrich/Gstrich


fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0.0, 1.5)
ax.set_ylim(0.0, 45)
print(1/gamma)
print(y)

data = [[value.n for value in 1/gamma], [value.n for value in 1/gammastrich]]
labels = [ ["$y\\left(\\frac{1}{\\gamma}\\right)$-Messwerte", "Fit zu $y\\left(\\frac{1}{\\gamma}\\right)$: $y = f \\cdot \\left(1+\\frac{1}{\\gamma}\\right) + c $"],   
           ["$y'\\left(\\frac{1}{\\gamma'}\\right)$-Messwerte", "Fit zu $y'\\left(\\frac{1}{\\gamma'}\\right)$: $y' = f \\cdot \\left(1+\\frac{1}{\\gamma'}\\right) + c $"]  ]
colors = [['mediumblue', 'cornflowerblue'], ['darkred', 'tomato']]


y_data = np.array([value.nominal_value for value in y])
y_err = np.array([value.s for value in y])


for i in range(0,2,1):
    #Daten
    x_data = data[i]

    #Messwerte plotten
    ax.errorbar(x_data, y_data, yerr=y_err, label=labels[i][0]  , color = colors[i][0], linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2 )

    # Fitfunktion definieren
    def fit_function(x, f, c):
        return f * (1+x) + c

    # Curve-Fit mit Unsicherheiten in y
    params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
    f_value = params[0]
    c_value = params[1]
    fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
    f_error = fit_errors[0]
    c_error = fit_errors[1]

    dof = len(RF.index)-len(params)
    chi2 = sum([((fit_function(x, f_value, c_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

    # Fit-Ergebnisse ausgeben
    #print(f"A = {A_value:.6f} ± {A_error:.6f}")
    #print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
    #print(f"Chi-Quadrat/dof: {chi2/dof}")

    x_ax = np.linspace(0, 10, 1000) 
    y_ax = fit_function(x_ax, f_value, c_value)

    labels[i][1] = labels[i][1] + f"\n $f = {f_value:.6f} \\pm {f_error:.6f}$\n $c = {c_value:.6f} \\pm {c_error:.6f}$"
    # Plot zeichnen
    plt.plot(x_ax, y_ax, label=labels[i][1], linewidth=2, color=colors[i][1])






plt.xlabel("$\\frac{1}{\\gamma}$",fontsize=fnt)
plt.ylabel("$y$ in cm", fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Entfernung $y$ zwischen Kante und Gegenstand in Abhängigkeit von $\\frac{1}{\\gamma}$", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O10/Brennweite.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()

