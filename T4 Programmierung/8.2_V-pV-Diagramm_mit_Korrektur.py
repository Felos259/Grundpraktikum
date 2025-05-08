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

fnt = 25 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]


RF = pd.read_csv('T4 Programmierung/BoyleMariotte.csv', header=3)
RG = pd.read_csv('T4 Programmierung/Totvolumen.csv', header=0)

uL = unp.uarray(RF['Laenge'], RF['dL'])
up =unp.uarray(RF['Druck'], RF['dp'])
uTot=unp.uarray(RG['Totvolumen'],RG['dTotvolumen'])

# Volumen bestimmen
uv = uL*(np.pi*unp.uarray([2.5],[0.005])**2)-uTot #Volumen (inkl. Unsicherheit) - berechnetes Totvolumen
RF['volumen']=np.array([value.nominal_value for value in uv])
RF['delvolumen']=np.array([value.s for value in uv])

# Berechnung von p*V in Joule
upV = up*uv
RF['pV']=np.array([value.nominal_value for value in upV])/10
RF['delpV']=np.array([value.s for value in upV])/10

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,400)
ax.set_ylim(0.001, 40)

# Plot der Messwerte V und p*V mit Errorbars 
ax.errorbar(RF.volumen, RF.pV, xerr=RF.delvolumen , yerr=RF.delpV, label='$p(V)$', color = '#339999', linestyle='None',  marker='o', markersize=9, capsize=6,elinewidth =1.5)

# linearer Fit

# Fitfunktion definieren
def fit_function(x, A, x0):
    return A * x + x0

#Daten
x_data = RF['volumen']
x_err = RF['delvolumen']
y_data = RF['pV']
y_err = RF['delpV'] 

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True)
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
x0_value = params [1]
x0_error = fit_errors[1]

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,A_value,x0_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0, 500, 1000) 
y_ax = fit_function(x_ax, A_value,x0_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = A \\cdot x+x_0$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$ \n $x_0 = {x0_value: .6f} \\pm {x0_error: .6f}$ ", linewidth=2, color='#a83e9e')

plt.xlabel('Volumen $V$ (inklusive theoretischem Totvolumen) in cm$^3$', fontsize=fnt)
plt.ylabel("$V \cdot p(V)$ in Joule", fontsize=fnt)
plt.legend(loc = 'lower right', fontsize=fnt)
plt.grid()
ax.set_facecolor("#f9f9f9")
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)  

plt.title("$V$-$V\cdot p$-Diagramm", fontsize=fnt)

plt.savefig("T4 Programmierung/pVDiagramm_korr.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
# plt.savefig("T4 Programmierung/pVDiagramm_korr.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

print("Mittelwert von p*V:", sum(upV)/len(upV))

# Berechnung der verwendeten Stoffmenge
temperature = unp.uarray([23.8+273.15],[0.3])
R = 8.31446261815324
print("Stoffmenge: ",unp.uarray([A_value],[A_error])/(temperature*R*(100**3)))


# Es fehlt noch das Teilen durch Temperatur*R, um das korrekte Ergebnis zu bestimmen.


# plt.show()




