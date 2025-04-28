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

# WICHTIGE ÄNDERUNGEN ZU MACHEN: Unsicherheit L und M fixen
# QUADRIERTE y-ACHSE - F_0-f^2-Diagramme 

#################################################

# Rohdaten (Anregefrequenzen), aus denen die Resonanzfrequenzen berechnet werden 
RF = pd.read_csv('M12/Resonanzfrequenzen.csv', header=2, sep=';')

# Frequenz der Saite = 2*Anregefrequenz

for column_name, column in RF.items(): 
    #column ist das Array, das zu Zeile column_name gehört
    if column_name != "deltaF":
        for i in range (0,len(column),1):
            column[i] = column[i]*2

# Index renamen damit er bei 1 anfängt
RF.index = np.arange(1, len(RF) + 1)

Frq = {'Reihe1' : unp.uarray(RF.Reihe1, RF.deltaF), 
       'Reihe2' : unp.uarray(RF.Reihe2, RF.deltaF),
       'Reihe3' : unp.uarray(RF.Reihe3, RF.deltaF)}

L = u.ufloat(0.6,0.008)
dM = 0.007
M = unp.uarray([1, 0.5, 0.5], [dM, dM, dM])

##################################################################

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Datenträger, damit man subplots ansprechen kann
lines = []

# Achsen richten
ax.set_xlim(0.0000001,8.5)
ax.set_ylim(0, 2000)



# Plot der Messwerte mit Errorbars 
lines += ax.errorbar(RF.index, RF.Reihe1, yerr=RF.deltaF, label='Reihe 1: 60cm, 1000g, 3. Kerbe', color = 'lightblue', linestyle='None', marker='o', capsize=6)
lines += ax.errorbar(RF.index, RF.Reihe2, yerr=RF.deltaF, label='Reihe 2: 60cm, 500g, 3. Kerbe', color = 'limegreen', linestyle='None', marker='o', capsize=6)
lines += ax.errorbar(RF.index, RF.Reihe3, yerr=RF.deltaF, label='Reihe 3: 60cm, 500g, 1. Kerbe', color = 'coral', linestyle='None', marker='o', capsize=6)
err_legends= ['Reihe 1: 60cm, 1000g, 3. Kerbe', 'Reihe 2: 60cm, 500g, 3. Kerbe', 'Reihe 3: 60cm, 500g, 1. Kerbe',' ' ,' ' ,' ',' ',' ',' ']

#########################################################

# Linearer Fit der Messergebisse

# Fitfunktion definieren
def fit_function(x, A):
    return A * x

# x-Achse erstellen, an der der Plot angezeichnet werden kann
x_ax=np.linspace(0, 10, 1000) 

# Farben und Legenden definieren
colors = {'Reihe1':'blue', 'Reihe2':'lightgreen', 'Reihe3':'plum'}
legends = {'Reihe1':'Fit Reihe 1: ', 'Reihe2':'Fit Reihe 2: ', 'Reihe3':'Fit Reihe 3: '}

for column_name, column in RF.items(): 
    #column ist das Array, das zu Zeile column_name gehört
    if column_name != "deltaF":
        #Daten
        x_data = RF.index
        y_data = column
        y_err = RF.deltaF 

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
        #print(f"Chi-Quadrat/dof: {chi2/dof}")

        y_ax = fit_function(x_ax, A_value)
        legends[column_name] = legends[column_name] + f"$y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$"

        # Eigentlichen Plot zeichnen
        lines += ax.plot(x_ax, y_ax, 
                 linewidth=2, color=colors[column_name])
         
        
        
# Legenden aufsplitten und hübsch platzieren
ax.legend([lines[0], lines[3], lines[6]], err_legends,
          loc='lower right')

from matplotlib.legend import Legend

leg = Legend(ax, lines[9:], [legends['Reihe1'], legends['Reihe2'], legends['Reihe3']],
             loc='upper left')
ax.add_artist(leg);

#print('ellllloEEEEEEEEEEEEEEEEEEEEEEEEEEEE#',ax.get_legend_handles_labels())

# Hübschigkeiten
plt.xlabel('n')
plt.ylabel("Frequenz $f$ in $\\frac{1}{s}$")
plt.title("Resonanzfrequenzen in Abhängigkeit von n")


plt.savefig("Resonanzfrequenzen.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.savefig("Resonanzfrequenzen.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 
# funktioniert mit allen Datentypen - vorher Datei erstellen
# 1. Zeile, um Bild zu checken, 2. für LaTex

#plt.show() 

########################################

# Phasengeschwindigkeit c berechnen - 
# c = (f_n*2*L) / n

# Irgendwelche dummy-Werte einsetzen
Frq['c1'] = Frq['Reihe1']
Frq['c2'] = Frq['Reihe1']
Frq['c3'] = Frq['Reihe1']

for i in range(1, len(column), 1):
    Frq['c1'][i] = (RF.Reihe1[i]*2.0*L)/i
    Frq['c2'][i] = (RF.Reihe2[i]*2.0*L)/i
    Frq['c3'][i] = (RF.Reihe3[i]*2.0*L)/i

########################################

# Lineare Massendichte mü = m / L

mu = M / L


print(mu)