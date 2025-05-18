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

# Systematische Unsicherheit Durchmesser Kapillare
dd = 0.0

# DICHTE DESTILIERTES WASSER WIRD ALS 1000 kg/m^3 ANGENOMMEN
rho = 1000

# Gravitationskonstante
# https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html
ug = u.ufloat(9.812669, 0.000011)

# Durchmesser der Kapillaren bestimmen
Durchmesser = pd.read_csv('M5/Kapillare/DurchmesserKapillaren.csv', header=0, sep=';')

# Farben Error- und Fitdiagramm
colors = [['mediumblue', 'cornflowerblue'],
          ['forestgreen', 'greenyellow'],
          ['firebrick', 'tomato'],
          ['chocolate', 'darkorange']]

dmean = unp.uarray([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
hmean = unp.uarray([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-50, 50)

# Plot mit den gemessenen Steighöhen (Index-h-Diagramm)
for i in range(0,4,1):
    # Durchmesser der Kapillare berechenen
    dmean[i] = np.mean(unp.uarray(Durchmesser['d'+str(i+1)], dd))

    # Steighöhe der Kapillare einlesen
    Steigh = pd.read_csv('M5/Kapillare/Steighohe'+str(i+1)+'.csv',  header=0, sep=';')

    # Mittelwerte der Steighöhen einlesen 
    hmean[i] = np.mean(unp.uarray(Steigh['h'], Steigh['dh']))
    h = unp.uarray(Steigh['h'], Steigh['dh'])

    # Index renamen damit er bei 1 anfängt
    Steigh.index = np.arange(1, len(Steigh) + 1)

    #Daten
    x_data = Steigh.index
    y_data = np.array([value.nominal_value for value in h])
    y_err = np.array([value.s for value in h])

    #Messwerte plotten
    ax.errorbar(x_data, y_data, yerr=y_err, label='Kapillare mit d $\\cong$'+str(dmean[i]) , color=colors[i][0], linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=1.5 )
    # Horizontale Linie bei y=Mittelwert h
    plt.axhline(hmean[i].n, color=colors[i][0], linewidth=0.8, linestyle='--')  


plt.xlabel('n',fontsize=fnt)
plt.ylabel('Steighöhe $h$ in m', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Steighöhe der Kapillaren", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("M5/Kapillare/Index-h-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()
#
# Plot der einzelnen Messungen
#

#####################################################################
#
# Plot der Mittelwerte
#

durchr = 1/(0.5*dmean)

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 6.5)
ax.set_ylim(0, 700)

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
#Daten
x_data = np.array([value.n for value in durchr])
x_err = np.array([value.s for value in durchr])
y_data = np.array([value.n for value in hmean])
y_err = np.array([value.s for value in hmean])


# Plot der Messwerte mti Errorbars
ax.errorbar( x_data, y_data, xerr=x_err, yerr=y_err, label='$mittlere Steighöhen$', color = colors[0][1], linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=1.5)

# linearer Fit
# Fitfunktion definieren
def fit_function(x, A):
    return A * x

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]

dof = len(x_data.index)-len(params)
chi2 = sum([((fit_function(x,A_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax = np.linspace(0, 10, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.xlabel('$\\frac{1}{r}$ in m$^-1$',fontsize=fnt)
plt.ylabel('mittlere Steighöhe $\\bar{h}$ in m', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Mittelwerte der Steighöhen der Kapillaren $\\bar{h}$ in Abhängigkeit des Radius $r$", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("M5/Kapillare/durchr-h-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()

uA = u.ufloat(A_value, A_error)

sigma = uA * rho * ug * 0.5
print('Sigma = ' + sigma)