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

#WICHTIGE ÄNDERUNGEN ZU MACHEN: Unsicherheiten fixen

RF = pd.read_csv('GrundfrequenzSpannung.csv', header=3)

uL = u.ufloat(0.6, 0.0005)

dM = 0.000005
Massen = unp.uarray([0.5, 0.2, 0.050], [dM, dM, dM])
uM = []

for j in range(0, len(RF['Masse']), 1):
    if RF['Masse'][j] == 1:
        uM.append(Massen[0] + 2*Massen[1] + 2*Massen[2])
    elif RF['Masse'][j] == 0.5:
        uM.append(2*Massen[1] + 2*Massen[2])
    elif RF['Masse'][j] == 0.25:
        uM.append(Massen[1]+Massen[2])

dM = np.array([value.s for value in uM])


uM = unp.uarray(RF['Masse'], dM)

# Massendichte mü berechnen
umu = uM / uL 

# Zugspannungskraft F_0 berechnen - F_0 = i*M*g + F_LH, i = Kerbe

F_LH = 0.52 # in Anleitung gegeben
g = 9.812669 #https://www.ptb.de/cms/ptb/fachabteilungen/abt1/fb-11/fb-11-sis/g-extractor.html

F_0 = RF['Kerbe'] * uM * g + F_LH

# Frequenz der Saite = 2 * Anregefrequenz, Unsicherheiten berechnen
for j in range(0, len(RF['df']) , 1):
    RF.loc[j, 'Grundfrequenz'] = RF['Grundfrequenz'][j]*2
    if(RF['Grundfrequenz'][j]<=100):
        RF.loc[j,'df'] = (0.0001 * RF['Grundfrequenz'][j] + 0.02)
    else:
        RF.loc[j,'df'] = (0.0001 * RF['Grundfrequenz'][j] + 0.2)
    
uf = unp.uarray(RF['Grundfrequenz'], RF['df'])
ufsqrd = uf**2

# Wert und Unsichheit in Dataset einlesen
RF['F_0']=np.array([value.nominal_value for value in F_0])
RF['deltaF_0']=np.array([value.s for value in F_0])
RF['fQuad']=np.array([value.nominal_value for value in ufsqrd])
RF['deltafQuad']=np.array([value.s for value in ufsqrd])


#Daten
x_data = RF['F_0']
y_data = RF['fQuad']
x_err = RF['deltaF_0']
y_err = RF['deltafQuad'] 

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,20.999)
ax.set_ylim(0,25000)

# Plot der Messwerte L und 1/f mit Errorbars 
ax.errorbar(x_data, y_data, xerr=x_err , yerr=y_err, label='$f^2$ in Abhängigkeit der Zugspannung $F_0$', 
            color = 'lightblue', linestyle='None', marker='o', capsize=6)

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
chi2 = sum([(fit_function(x,A_value)-y)*2/u*2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 25, 1000) 
y_ax = fit_function(x_ax, A_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$", linewidth=2, color='blue')

plt.xlabel('Zugspannung $F_0$ in N')
plt.ylabel("$f^2$ in Hz$^2$")
plt.legend()
plt.title("$F_0$-$f^2$-Diagramm")

plt.savefig("FrequenzSpannung.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 
#plt.savefig("M12/FrequenzSpannung.svg", format='svg', bbox_inches='tight', pad_inches=0.1) 

plt.show() 

########################################

# keine Ahnung was man beim funktionalen Zusammenhang machen muss

########################################

# mu berechnen
mu = F_0/(ufsqrd*4*uL**2)

# mu-Werte von anderem Versuch einfügen
mu2 = pd.read_csv('mu.csv', header=0, sep=';')
mu2 = unp.uarray(mu2['mu'], mu2['deltaMu'])

mu = np.concatenate([mu2, mu], axis=0)
