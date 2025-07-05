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

RF = pd.read_csv('O8/Gitter.csv', header=0, sep=';')

# Absstand zwischen Schirm und Spalt
SS = u.ufloat(37.0, 0.5) - u.ufloat(7.2, 0.5) 
lamb = 532 * 10**-6

# Position der Minima
position = unp.uarray(RF['position'], RF['dPos'])


# Index renamen damit die Ordnungen richtig liegen
RF.index =  np.arange(-1 * RF.idxmax()[2], len(RF) - RF.idxmax()[2])
# RF.idxmax()[2] = Index an dem Intensity maximal ist => Index, welches Macima der Ordnung 0 - ist

# Positionen so verschieben, dass 0 cm zwischen den Minimas der Ordnung 1 liegt
position = (position - RF['position'][0])

print(RF['position'][0])

print(position)

# Kleiwinkelnäherung sin(alp)=tan(alp)= pos/SS
sinAlph = position/SS

print(sinAlph)

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achse richten
ax.set_xlim(-0.2, 3.3)
ax.set_ylim(-0.03, 0.2)


#Daten
x_data = abs(RF.index)
y_data = np.array([abs(value.n) for value in sinAlph])
y_err = np.array([abs(value.s) for value in sinAlph])

#Messwerte plotten
ax.errorbar(x_data, y_data, yerr=y_err, label = 'Messwerte', 
            color = 'mediumblue', linestyle='None', marker='o', capsize=10, markersize=7, elinewidth=2)


# linearer Fit
# Fitfunktion definieren
def fit_function(x, A):
    return  x * lamb / A

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,A_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax = np.linspace(-20, 20, 1000) 
y_ax = fit_function(x_ax, A_value)

label = r'Fit: $\sin(\alpha) = n \cdot \frac{\lambda}{g}$' + f"\n $(g = {A_value:.6f} \\pm {A_error:.6f})$ cm"

# Plot zeichnen
plt.plot(x_ax, y_ax, label = label, linewidth = 2, color = 'lightblue')


################

# cosmetics

plt.xlabel('Ordnung n des Minimas',fontsize=fnt)
plt.ylabel('$\\sin{\\alpha}$', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Abhänigkeit der Position eines Maximas von seiner Ordnung n", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)
plt.grid()
plt.savefig("O8/GitterOrdnungen.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()


###########################
# Abbe-Kriterium nachvollziehen

