import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
import uncertainties as u
import uncertainties.umath as um
from uncertainties import unumpy as unp
import csv

# Ausgangspunkt der Messung in MILLIMETERN
a_0 = u.ufloat(0.0, 0.0)
# Gewicht des Bügels in KILOGRAMM
um = u.ufloat(0.0, 0.0)
# Länge des Messdrahtes in METERN
ul = u.ufloat(0.0, 0.0)

# Gravitationskonstante
# https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html
ug = u.ufloat(9.812669, 0.000011)

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

RF = pd.read_csv('M5/Bugelmethode/Kalibrierung.csv', header=0, sep=';')
## l un dl in mm!!!


# l abspeichern und Kalibrierungs l' berechnen
ul = unp.uarray(RF['l'], RF['dl'])
ulStrich = (ul-a_0)*10**-3

aF = 0.1 * RF['Kerbe'] * um * ug

# Massen-Auslenkungsdiagramm (M-L)

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
#Daten
x_data = np.array([value.nominal_value for value in aF])
x_err = np.array([value.s for value in aF])
y_data = np.array([value.nominal_value for value in ulStrich])
y_err = np.array([value.s for value in ulStrich])


fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 1.25)
ax.set_ylim(0, 2.75)

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar( x_data, y_data, xerr=x_err, yerr=y_err, label='$B(I)$', color = 'lightblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=1.5)

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
chi2 = sum([((fit_function(x,A_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax = np.linspace(0, 10, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Auslenkungskraft $F$ in N', fontsize=fnt)
plt.ylabel('Auslenkung $a\'$ in m', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Auslenkung der Federwaage in Abhängigkeit der Belastung", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("M5/Bugelmethode/Kalibrierungsdiagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 

plt.show()

df = {'A': A_value, 'dA': A_error}

with open('M5/KaliFaktoren.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(df)

uA = u.ufloat(A_value, A_error)

# Werte Nahe Null
NN = pd.read_csv('M5/Bugelmethode/Bugelmethode0.csv', header=0, sep=';')
# Werte ZimmerTemperatur
ZT = pd.read_csv('M5/Bugelmethode/Bugelmethode20.csv', header=0, sep=';')

# Werte abspeichern
a0NN = unp.uarray(NN['a_0'], NN['da_0'])
aAbrissNN = unp.uarray(NN['abrissA'], NN['dabrissA'])
a0ZT = unp.uarray(ZT['a_0'], ZT['da_0'])
aAbrissZT = unp.uarray(ZT['abrissA'], ZT['dabrissA'])

#Mittelwerte berechnen
aBarNN = np.mean(aAbrissNN - a0NN)
aBarZT = np.mean(aAbrissZT - a0ZT)

# Oberflächenspannung aus Kalibrierungskruve berechnen
sigmaNN = aBarNN/(2*uA*ul)
print('Sigma nahe Null =' + sigmaNN)

sigmaZT = aBarZT/(2*uA*ul)
print('Sigma Zimmertemperatur =' + sigmaZT)