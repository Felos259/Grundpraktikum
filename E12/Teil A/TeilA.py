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

# Unscicherheiten anpassen k

mu_0 = 4*np.pi*10**(-7)
N = 320
R = 0.068

k = mu_0 *N/(2*R)*(4/5)**(3/2)

# Anodenspannungen mit Unsicherheiten
uU_A = unp.uarray([3000, 4000, 5000], [100, 100, 100]) 

################################

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 7)
ax.set_ylim(0, 1)

# Datenträger, damit man subplots ansprechen kann - für hübsche Legende
# Labels und Farben für Achsen speichern 
label = {'E12/Teil A/3kV.csv' : ['$U_A=3$kV', 'Fit zu $U_A=3$kV', 'lightblue' ,'blue', 'E12/Teil A/3kVCOPY.csv'],
         'E12/Teil A/4kV.csv' : ['$U_A=4$kV', 'Fit zu $U_A=4$kV', 'coral', 'red', 'E12/Teil A/4kVCOPY.csv'],
         'E12/Teil A/5kV.csv' : ['$U_A=5$kV', 'Fit zu $U_A=5$kV', 'limegreen', 'green', 'E12/Teil A/5kVCOPY.csv']}

# Abspeichern der FIT-Wert für Berechnungen von e/m
uA = [] 

for column_name, column in label.items():

    RF = pd.read_csv(column_name, header=0, sep=';')

    for j in range(0, len(RF['dI']) , 1):
        # Einheiten in SI umwandeln
        RF.loc[j, 'I']  = RF['I'][j] * 10**-1
        RF.loc[j, 'g']  = RF['g'][j] * 10**-3
        RF.loc[j, 'dg'] = RF['dg'][j] * 10**-3
        
        # Unsicherheiten in I Berechnen
        RF.loc[j, 'dI'] = 0.012 * RF['I'][j] + 0.005

    ug = unp.uarray(RF['g'], RF['dg'])

    ue = 0.08 - ug # e ist Schirmlänge Minus gemessener Wert, haben eigentlich g gemessen
    RF['e'] = np.array([value.nominal_value for value in ue])
    RF['de'] = np.array([value.s for value in ue])

    uI = unp.uarray(RF['I'], RF['dI'])
    KL = u.ufloat(0.08, 0.0005) # Kantenlänge Schirm
    # 1/I für antiproportionalen Zusammenhang berechnen
    
    UdurchI = 1/uI
    RF['durchI'] = np.array([value.nominal_value for value in UdurchI])
    RF['DelDurchI'] = np.array([value.s for value in UdurchI])
    #aus den Messwerten Magnetfeldstärke berechnen
    #uB = k*uI

    #aus den Messwerten r(I) berechnen
    ur = (KL**2+ue**2)/(np.sqrt(2)*(KL-ue))

    RF['r'] = np.array([value.nominal_value for value in ur])
    RF['dr'] = np.array([value.s for value in ur])

    header = ["I", 'dI' , 'durchI', 'DelDurchI', 'g', 'dg', "e", 'de', 'r', 'dr']

    RF.to_csv(column[4], sep='&', columns = header, index = False)

    #Daten
    x_data = np.array([value.nominal_value for value in UdurchI])
    x_err = np.array([value.s for value in UdurchI])
    y_data = np.array([value.nominal_value for value in ur])
    y_err = np.array([value.s for value in ur])

    #Messwerte plotten
    ax.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, label=column[0], color=column[2], linestyle='None', marker='o', capsize=8, markersize=8, elinewidth=1.5)

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

    x_ax = np.linspace(0, 1000, 1000) 
    y_ax = fit_function(x_ax, A_value)

    column[1] = column[1] + f"$y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$"

    uA.append(u.ufloat(A_value, A_error))
    
    # Plot zeichnen
    plt.plot(x_ax, y_ax, label=column[1], linewidth=2, color=column[3])

plt.xlabel('$\\frac{1}{I}$ in $\\frac{1}{A}$', fontsize=fnt)
plt.ylabel('Radius $r$ in m', fontsize=fnt)
plt.legend(fontsize=fnt) #Legende printen
plt.title("Radius $r$ in Abhängigkeit der Stromstärke $I$", fontsize=fnt)

plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("E12/Teil A/r-1durchI-Diagramm.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("E12/Teil A/r-1durchI-Diagramm.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

plt.show() 

# uA zu schönerem Datentyp machen
uA = unp.uarray([value.nominal_value for value in uA],
                [value.s for value in uA]) 

# e/m berechnen
ue_m = 2*uU_A / (k * uA)**2
print(ue_m * 10**-11)