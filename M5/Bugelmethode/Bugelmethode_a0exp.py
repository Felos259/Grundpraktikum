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
fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

### Grundzustand der Kalibrierungsmessung per Fit bestimmen



# Gewicht des Bügels in KILOGRAMM
um = u.ufloat(1.293, 0.001)*10**-3
# Länge des Messdrahtes in METERN
uDrahtlaenge = u.ufloat(0.0505, 0.0005)
# Gravitationskonstante
# https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html
ug = u.ufloat(9.812669, 0.000011)


RF = pd.read_csv('M5/Bugelmethode/Kalibrierung.csv', header=1, sep=';')
## l un dl in mm!!!


# l abspeichern und Kalibrierungs l' berechnen
ul = unp.uarray(RF['l'], RF['dl'])  # UNSICHERHEIT IST ABSOLUT!!!
ulStrich = (ul)*10**-3

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
ax.set_xlim(0.0, 0.013)
ax.set_ylim(0.0000000001, 0.015)

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar( x_data[0:9], y_data[0:9], xerr=x_err[0:9], yerr=y_err[0:9], label='$a\'_i(F)$ - i = 1,..., 9', color = 'cornflowerblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=1.5)
# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar( x_data[9:18], y_data[9:18], xerr=x_err[9:18], yerr=y_err[9:18], label='$a\'_i(F)$ - i = 9, ..., 1', color = 'lightblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=1.5)

# linearer Fit
# Fitfunktion definieren
def fit_function(x, A, x_0):
    return A * x + x_0

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
x0_value = params[1]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
x0_error = fit_errors[1]

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,A_value, x0_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax = np.linspace(0, 10, 1000) 
y_ax = fit_function(x_ax, A_value, x0_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$, \n $ x_0 = {x0_value:.6f} \\pm {x0_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Auslenkungskraft $F$ in N', fontsize=fnt)
plt.ylabel('Auslenkung $a\'$ in m', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Auslenkung der Federwaage in Abhängigkeit der Belastung", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("M5/Bugelmethode/Kalibrierungsdiagramm_experimentellesA0.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 

plt.show()
 
# ------------------ 

uA = u.ufloat(A_value, A_error)

# Werte Nahe Null
NN = pd.read_csv('M5/Bugelmethode/Bugelmethode0.csv', header=1, sep=';')
# Werte ZimmerTemperatur
ZT = pd.read_csv('M5/Bugelmethode/Bugelmethode20.csv', header=0, sep=';')


# Mittelwert der 2.-6. Messung berechnen, um einen Nullpunkt für die Messung nahe Null zu finden (Messmethode zuerst falsch angewendet)
a0mean_NN = np.mean(NN['a_0'][1:])
print('MEAN A0 NN: ', a0mean_NN)

# korrigierte Standardabweichung berechnen
std_NN = np.std(NN['a_0'][1:], ddof=1)
print('STD A0 NN: ', std_NN)

deltaStd_NN = std_NN * np.sqrt(5)
print('DELTA STD A0 NN: ', deltaStd_NN)

deltaa_0Bar_NN = np.sqrt(deltaStd_NN**2+0.2**2)
print('DELTA a0BAR A0 NN: ', deltaa_0Bar_NN, '\n')

# Mittelwert abspeichern
a0NN = u.ufloat(a0mean_NN, deltaa_0Bar_NN)

# Abrisswerte abspeichern
aAbrissNN = unp.uarray(NN['abrissA'], NN['dabrissA']) - a0NN
aAbrissZT = unp.uarray(ZT['abrissA'], ZT['dabrissA']) - a_0


#Mittelwert des Abrisses bei Temperaturen Nahe Null
aBarNN = np.mean([value.nominal_value for value in aAbrissNN])
print('MEAN NN: ', aBarNN)

# Unsicherheit Mittelwert NN
std_NN = np.std([value.nominal_value for value in aAbrissNN], ddof=1)
print('STD NN: ', std_NN)

deltaStd_NN = std_NN * np.sqrt(6)
print('DELTA STD NN: ', deltaStd_NN)

deltaa_0Bar_NN = np.sqrt(deltaStd_NN**2+0.4**2)
print('DELTA FBAR NN: ', deltaa_0Bar_NN, '\n')

aBarNN = u.ufloat(aBarNN, deltaa_0Bar_NN)


# Mittelwert des Abrisses bei Zimmertemperatur
aBarZT = np.mean([value.nominal_value for value in aAbrissZT])
print('MEAN ZT: ', aBarZT)

# Unsicherheit Mittelwert ZT
std_ZT = np.std([value.nominal_value for value in aAbrissZT], ddof=1)
print('STD ZT: ', std_ZT)

deltaStd_ZT = std_ZT * np.sqrt(6)
print('DELTA STD NN: ', deltaStd_ZT)

deltaa_0Bar_ZT = np.sqrt(deltaStd_ZT**2+0.4**2)
print('DELTA FBAR NN: ', deltaa_0Bar_ZT, '\n')

aBarZT = u.ufloat(aBarZT, deltaa_0Bar_ZT)

# Oberflächenspannung aus Kalibrierungskruve berechnen
sigmaNN = aBarNN/(2*uA*uDrahtlaenge)
print('Sigma nahe Null =', sigmaNN)
print(sigmaNN.s, '\n')

sigmaZT = aBarZT/(2*uA*uDrahtlaenge)
print('Sigma Zimmertemperatur =', sigmaZT)
print(sigmaZT.s)