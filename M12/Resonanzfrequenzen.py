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

# WICHTIGE ÄNDERUNGEN ZU MACHEN: Unsicherheit L und M fixen
# QUADRIERTE y-ACHSE - F_0-f^2-Diagramme 

#################################################

# Rohdaten (Anregefrequenzen), aus denen die Resonanzfrequenzen berechnet werden 
RF = pd.read_csv('M12/Resonanzfrequenzen.csv', header=2, sep=';')

dM = 0.000005
Massen = unp.uarray([0.5, 0.2, 0.050], [dM, dM, dM])
uM = []
uM.append(Massen[0] + 2*Massen[1] + 2*Massen[2])
uM.append(2*Massen[1] + 2*Massen[2])
uM.append(2*Massen[1] + 2*Massen[2])

dM = np.array([value.s for value in uM])
uM = unp.uarray([1, 0.5, 0.5], dM)

# Frequenz der Saite = 2 * Anregefrequenz, Unsicherheiten berechnen
cols = ['Reihe1', 'Reihe2', 'Reihe3']
unsicherCols = ['delta1', 'delta2', 'delta3']

for i in range(0,3,1): 
    for j in range(0, len(RF[cols[i]]) , 1):
        RF.loc[j, cols[i]] = RF[cols[i]][j]*2
        if(RF[cols[i]][j]<=100):
            RF.loc[j,unsicherCols[i]] = 0.0001 * RF[cols[i]][j] + 0.02 # np.sqrt((0.0001 * RF[cols[i]][j] + 0.02)**2 + 1**2)
        else:
            RF.loc[j,unsicherCols[i]] = (0.0001 * RF[cols[i]][j] + 0.2) # np.sqrt((0.0001 * RF[cols[i]][j] + 0.2)**2 + 1**2)
# Index renamen damit er bei 1 anfängt
RF.index = np.arange(1, len(RF) + 1)

uFrq = {'Reihe1' : unp.uarray(RF.Reihe1, RF.delta1), 
       'Reihe2' : unp.uarray(RF.Reihe2, RF.delta2),
       'Reihe3' : unp.uarray(RF.Reihe3, RF.delta3)}

uL = u.ufloat(0.6,0.0005)

# RF.to_csv('M12/copyReson.csv', sep='&')

##################################################################

# Farben und Legenden definieren
colors = ['lightblue', 'lightgreen', 'pink', 'blue', 'green', 'red']
legends = ['Reihe 1: 60cm, 1000g, 3. Kerbe','Reihe 2: 60cm, 500g, 3. Kerbe', 'Reihe 3: 60cm, 500g, 1. Kerbe', 'Fit Reihe 1: ', 'Fit Reihe 2: ', 'Fit Reihe 3: ']

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Datenträger, damit man subplots ansprechen kann
lines = []

# Achsen richten
ax.set_xlim(0.0000001,8.5)
ax.set_ylim(0, 1500)

# Plot der Messwerte mit Errorbars 
lines += ax.errorbar(RF.index, RF.Reihe1, yerr=RF.delta1, label=legends[0], color=colors[0] , linestyle='None', marker='o', capsize=6)
lines += ax.errorbar(RF.index, RF.Reihe2, yerr=RF.delta2, label=legends[1], color=colors[1] , linestyle='None', marker='o', capsize=6)
lines += ax.errorbar(RF.index, RF.Reihe3, yerr=RF.delta3, label=legends[2], color=colors[2] , linestyle='None', marker='o', capsize=6)

#########################################################

# Linearer Fit der Messergebisse

# Fitfunktion definieren
def fit_function(x, A):
    return A * x

# x-Achse erstellen, an der der Plot angezeichnet werden kann
x_ax=np.linspace(0, 10, 1000) 

# Abspeichern der Fit-Wert für Berechnungen von e/m
A = []

for i in range(0,3,1): 
    
    #Daten
    x_data = RF.index
    y_data = RF[cols[i]]
    y_err = RF[unsicherCols[i]]
    
    # Curve-Fit mit Unsicherheiten in y
    params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
    A_value = params[0]
    fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
    A_error= fit_errors[0]

    dof = len(RF.index)-len(params)
    chi2 = sum([(fit_function(x,A_value)-y)*2/u*2 for x,y,u in zip(x_data,y_data,y_err)])

    # Fit-Ergebnisse ausgeben
    #print(f"A = {A_value:.6f} ± {A_error:.6f}")
    #print(f"x0 = {x0:.6f} ± {x0_error:.6f}")
    print(f"Chi-Quadrat/dof: {chi2/dof}")

    y_ax = fit_function(x_ax, A_value)
    legends[i+3] = legends[i+3] + f"$y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$"
    A.append(A_value)
    A.append(A_error)

    # Eigentlichen Plot zeichnen
    lines += ax.plot(x_ax, y_ax, 
                linewidth=2, color=colors[i+3], label=legends[i+3])
        
# Legenden aufsplitten und hübsch platzieren
# bei Errorbars hat man das Problem, dass Errorstriche und Markierungen je einen Platz in der Legende 'wegnehmen'
# err_legends= [legends[0], legends[1], legends[2],' ' ,' ' ,' ',' ',' ',' ']

# ax.legend([lines[0], lines[3], lines[6]], err_legends,
#           loc='lower right')

# from matplotlib.legend import Legend

# leg = Legend(ax, lines[9:], [legends[3], legends[4], legends[5]],
#              loc='upper left')
# ax.add_artist(leg);

# Hübschigkeiten
plt.xlabel('n')
plt.ylabel("Frequenz $f_n$ in Hz")
plt.title("Resonanzfrequenzen in Abhängigkeit von n")
plt.legend(prop={'size': 9})

plt.savefig("M12/Resonanzfrequenzen.pdf", format='pdf', pad_inches=0.1) 
plt.savefig("M12/Resonanzfrequenzen.svg", format='svg', pad_inches=0.1) 
# funktioniert mit allen Datentypen - vorher Datei erstellen
# 1. Zeile, um Bild zu checken, 2. für LaTex

# Residuenplot :)
residuen = [0,0,0]
for i in range(0,3,1):
    residuen[i] = RF[cols[i]] - A[2*i]*RF.index
fig, ax = plt.subplots()
ax.set_xlim(0,8.5)
ax.set_ylim(-6,13)

for i in range(0,3,1):
    plt.errorbar(x_data, residuen[i], fmt='o', label='Residuen', capsize=5, color=colors[i])
plt.tick_params(axis="both",direction="in",top=True,left=True,right=True,bottom=True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontale Linie bei y=0
plt.xlabel('n')
plt.ylabel('Residuen $(y_i - \hat{y}_i)$ in $Hz$')
plt.legend()
plt.title("Residuendarstellung zu den Resonanzfrequenzen in Abhängigkeit von n")

plt.savefig("ResiduumResonanzfrequenz.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 

plt.show() 

########################################

# Phasengeschwindigkeit c berechnen - 
# c = (f_n*2*L) / n

# Irgendwelche dummy-Werte einsetzen
uFrq['c1'] = uFrq['Reihe1']
uFrq['c2'] = uFrq['Reihe1']
uFrq['c3'] = uFrq['Reihe1']

for i in range(1, len(RF.index), 1):
    uFrq['c1'][i] = (uFrq['Reihe1'][i]*2.0*uL)/i
    uFrq['c2'][i] = (uFrq['Reihe2'][i]*2.0*uL)/i
    uFrq['c3'][i] = (uFrq['Reihe1'][i]*2.0*uL)/i

########################################

# Lineare Massendichte aus Messwerten

F_LH = 0.52 # in Anleitung gegeben
g = 9.812669 #https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html

Kerbe = [3,3,1]
F_0 = Kerbe * uM * g + F_LH
uA = unp.uarray([A[0], A[2], A[4]], [A[1], A[3], A[5]])

mu = F_0/(4*(uL*uA)**2)
c = 2 * uA * uL
#print(c)

df = [['mu','deltaMu']]

for value in mu:
    df.append([value.nominal_value, value.s])

with open('M12/Resonmu.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(df)
