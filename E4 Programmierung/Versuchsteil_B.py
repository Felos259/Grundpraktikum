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
import csv #Output meine Berechnungen in eine CSV-Datei

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

RF = pd.read_csv('E4 Programmierung/Versuchsteil_B.csv', header=1)

uf = unp.uarray(RF['Frequenz'], RF['df'])
uuch1 = unp.uarray(RF['U_Ch1'], RF['dU_Ch1'])
uuch2 = unp.uarray(RF['U_Ch2'], RF['dU_Ch2'])

# R_RL = Summe aus Omhschen Widerstand und Spulenwiderstand
widerstand = unp.uarray([10.16],[0.2016])+unp.uarray([151.02],[10*0.01+1.5102])

# Scheinwiderstand bestimmen
uscheinwiderstand = widerstand*(uuch1/uuch2)
RF['Z_RL2']=np.array([value.nominal_value for value in uscheinwiderstand])
RF['dZ_RL2']=np.array([value.s for value in uscheinwiderstand])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,600)
ax.set_ylim(0, 40000)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.Frequenz, RF.Z_RL2, xerr=RF.df , yerr=RF.dZ_RL2, label='$|Z_{RL}|$ in Abhängigkeit der Frequenz', color = 'lightblue', linestyle='None', marker='o', capsize=6)

# linearer Fit

# Fitfunktion definieren
def fit_function(x, A, x0):
    return np.sqrt((161.18)**2+(A * 2*np.pi*x)**2) #A entspricht der Induktivität der Spule, x0 entspricht R_RL

#Daten
x_data = RF['Frequenz']
x_err = RF['df']
y_data = RF['Z_RL2']
y_err = RF['dZ_RL2'] 

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
x0_value = params[1]
x0_error = fit_errors[1]

dof = len(RF.index)-len(params)
chi2 = sum([(fit_function(x,A_value,x0_value)-y)**2/u**2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 600, 1000) 
y_ax = fit_function(x_ax, A_value,x0_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = \\sqrt{{(2\\cdot \\pi \\cdot A \\cdot x)^2+x_0^2}}$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$ \n $x_0= {x0_value:.6f} \\pm {x0_error:.6f}$", linewidth=2, color='blue')
plt.xlabel('Frequenz $f$ in $s^{-1}$')
plt.ylabel("$|Z_{LR}|$ in $\\Omega$")
plt.legend()
plt.title("$f$-$|Z_{LR}|$-Diagramm")


# Inset
# Inset-Diagramm erstellen
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Position des Inset-Diagramms definieren
ax_inset = inset_axes(ax, width="30%", height="30%", loc=2, borderpad = 5, bbox_to_anchor=ax.bbox) #Positionierung mit bbox(x-Achse,y-Achse,Größe,Größe)

# Bereich für das Inset-Diagramm
x_inset = np.linspace(-0.18, 125.18, 1000)
y_inset = fit_function(x_inset, A_value,x0_value)

# Plotten im Inset-Diagramm
ax_inset.errorbar(RF.Frequenz, RF.Z_RL2, xerr=RF.df , yerr=RF.dZ_RL2, label='$|Z_{RL}|$ in Abhängigkeit der Frequenz', color = 'lightblue', linestyle='None', marker='o', capsize=6)
ax_inset.plot(x_inset, y_inset, linewidth=2)

# Inset-Bereich anpassen
ax_inset.set_xlim(-0.18, 125.18)
ax_inset.set_ylim(-0.18, 10000.18)
ax_inset.set_xticks([0, 100])
ax_inset.set_yticks([0, 2784.3,5000, 10000])
ax_inset.tick_params(labelsize=8)

# Induktivität L
print("Die Induktivität der Spule beträgt",unp.sqrt(unp.uarray([np.sqrt(A_value**2)],[np.sqrt(A_error**2)])),"H")

plt.savefig("VersuchsteilB.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("VersuchsteilB.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

plt.show()

# 2D list of variables (tabular data with rows and columns)
input_variable = [["Frequenz","df","Z_LR2","dZ_LR2"],[x_data,x_err,y_data,y_err]]
 
# Example.csv gets created in the current working directory
with open ('E4 Programmierung/WertetabelleVersuchsteilB.csv','w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerows(input_variable)



