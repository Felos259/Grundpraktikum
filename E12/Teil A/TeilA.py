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

# Unscicherheiten anpassen k, uU_A

mu_0 = 4*np.pi*10**(-7)
N = 320
R = 0.068

k = u.ufloat(mu_0 *N/(2*R)*(4/5)**(3/2),0)

################################


fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 7)
ax.set_ylim(0, 200)

# Datenträger, damit man subplots ansprechen kann - für hübsche Legende
# Labels und Farben für Achsen speichern 
label = {'E12/Teil A/3kV.csv' : ['$U_A=3$kV', 'Fit zu $U_A=3$kV', 'red', 'orange'],
         'E12/Teil A/4kV.csv' : ['$U_A=4$kV', 'Fit zu $U_A=4$kV', 'blue', 'purple'],
         'E12/Teil A/5kV.csv' : ['$U_A=5$kV', 'Fit zu $U_A=5$kV', 'grey', 'green']}

# Abspeichern der Fit-Wert für Berechnungen von e/m
uA = []

for column_name, column in label.items():
    
    RF = pd.read_csv(column_name, header=0, sep=';')

    ue = unp.uarray(RF['e'], RF['de'])
    uI = unp.uarray(RF['I'], RF['dI'])
    uU_A = unp.uarray(3000, 10) # Anodenspannung
    KL = u.ufloat(8, 0.05) # Kantenlänge Schirm

    # 1/I für antiproportionalen Zusammenhang berechnen
    UdurchI = 1/uI

    #aus den Messwerten Magnetfeldstärke berechnen
    #uB = k*uI

    #aus den Messwerten und der Magnetfeldstärke r(B) berechnen
    ur = (KL**2+ue**2)/(np.sqrt(2) * (KL-ue))

    #Daten
    x_data = np.array([value.nominal_value for value in UdurchI])
    x_err = np.array([value.s for value in UdurchI])
    y_data = np.array([value.nominal_value for value in ur])
    y_err = np.array([value.s for value in ur])

    #Messwerte plotten
    ax.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, label=column[0], color=column[2], linestyle='None', marker='o', capsize=6)

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

    column[2] = column[2] + f"$y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$"
    uA.append(u.ufloat(A_value, A_error))
    
    # Plot zeichnen
    plt.plot(x_ax, y_ax, label=column[1], linewidth=2, color=column[3])

plt.xlabel('$\\frac{1}{I}$ in $\\frac{1}{A}$')
plt.ylabel('Radius $r$ in cm')
plt.legend() #Legende printen
plt.title("Radius $r$ in Abhängigkeit der Stromstärke $I$")

plt.savefig("E12/Teil A/r-1durchI-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.savefig("E12/Teil A/r-1durchI-Diagramm.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

#
plt.show()
 

# uA zu schönerem Datentyp machen
uA = unp.uarray([value.nominal_value for value in uA],
                [value.s for value in uA]) 

# e/m berechnen
ue_m = 2 * uU_A / ((uA * k)**2)
print(ue_m)
