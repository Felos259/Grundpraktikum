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

RF = pd.read_csv('MessdatenO11.csv', header=1)

#To-Do: Unsicherheiten

RF['grundIntensitaet'] = 2400 #U_E in mV

ableseWinkel = 1 #Ableseunsicherheit Winkel in Grad
sysWinkel = 1 #Systematische Unsicherheit Winkel in Grad
RF['dWinkel'] = np.sqrt(ableseWinkel**2+sysWinkel**2)


ableseIntensitaet = 1 #Größtfehlerabschätzung der Intensität in mV
sysIntensitaet = 1 #Größtfehlerabschätzung der Intensität in mV
RF['unsicherheitIntensitaet'] = np.sqrt(ableseIntensitaet**2+sysIntensitaet**2)


# Berechnung Wurzel R für parallel polarisiertes Licht (PP) und senkrecht polarisiertes Licht (SP)
uRE = unp.uarray(RF['grundIntensitaet'],RF['unsicherheitIntensitaet'])
uRPP = unp.uarray(RF['intensePP'],RF['unsicherheitIntensitaet'])
uRSP = unp.uarray(RF['intenseSP'],RF['unsicherheitIntensitaet'])
print(uRPP)

uwRpp = unp.sqrt(uRPP/uRE)
uwRsp = unp.sqrt(uRSP/uRE)
RF['RPP']=np.array([value.nominal_value for value in uwRpp])
RF['dRPP']=np.array([value.s for value in uwRpp])
RF['RSP']=np.array([value.nominal_value for value in uwRsp])
RF['dRSP']=np.array([value.s for value in uwRsp])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,90)
ax.set_ylim(0,1)

#Daten
x_data = RF['winkel']/360*(2*np.pi)
x_err = RF['dWinkel']/360*(2*np.pi)

y_data_parallel = RF['RPP']
y_err_parallel = RF['dRPP']

y_data_senkrecht = RF['RSP']
y_err_senkrecht = RF['dRSP']


# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(x_data,y_data_parallel, xerr=x_err , yerr=y_err_parallel, label='Intensität des reflektierten Strahls für parallel polarisiertes Licht', color = 'darkblue', linestyle='None', marker='o', capsize=6)
ax.errorbar(x_data, y_data_senkrecht, xerr=x_err , yerr=y_err_senkrecht, label='Intensität des reflektierten Strahls für senkrecht polarisiertes Licht', color = 'darkgreen', linestyle='None', marker='o', capsize=6)

# Fitfunktion definieren
def fit_function_parallel(x, B):
    return ((np.tan(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.tan(x+np.arcsin(np.sin(x)/np.tan(B))))**2)

def fit_function_senkrecht(x,B):
    return ((np.sin(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.sin(x+np.arcsin(np.sin(x)/np.tan(B))))**2)


# Curve-Fit für parallel polarisiertes Licht
params, covariance = curve_fit(fit_function_parallel, x_data, y_data_parallel, sigma=y_err_parallel, absolute_sigma=True)
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value = params[0]
B_error = fit_errors[0]
#Chi^2/dof berechnen
dof = len(RF.index)-len(params)
chi2 = sum([((fit_function_parallel(x,B_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data_parallel,y_err_parallel)])
# Fit-Ergebnisse ausgeben
print(f"Parallel: B = {B_value:.6f} ± {B_error:.6f}")
print(f"Parallel: Chi-Quadrat/dof: {chi2/dof}")
# Fit-Ergebnisse plotten
x_ax=np.linspace(0, 300, 1000) 
y_ax_parallel = fit_function_parallel(x_ax, B_value)
plt.plot(x_ax, y_ax_parallel, label=f"Fit zu parallel polarisierten Licht mit Fitparameter \n $B = {B_value:.6f} \\pm {B_error:.6f}$", linewidth=2, color='blue')

# Curve-Fit für senkrecht polarisiertes Licht
params, covariance = curve_fit(fit_function_senkrecht, x_data, y_data_senkrecht, sigma=y_err_senkrecht, absolute_sigma=True)
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value = params[0]
B_error = fit_errors[0]
#Chi^2/dof berechnen
dof = len(RF.index)-len(params)
chi2 = sum([((fit_function_senkrecht(x,B_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data_parallel,y_err_parallel)])
# Fit-Ergebnisse ausgeben
print(f"Senkrecht: B = {B_value:.6f} ± {B_error:.6f}")
print(f"Senkrecht: Chi-Quadrat/dof: {chi2/dof}")
x_ax=np.linspace(0, 300, 1000) 
y_ax_senkrecht = fit_function_senkrecht(x_ax, B_value)
plt.plot(x_ax, y_ax_senkrecht, label=f"Fit zu senkrecht polarisierten Licht mit Fitparameter \n $B = {B_value:.6f} \\pm {B_error:.6f}$", linewidth=2, color='limegreen')

# Plot zeichnen
plt.xlabel('Eingangswinkel $\\alpha_E$ in Radiant')
plt.ylabel('$\\sqrt{R}$ in 1')
plt.legend()
plt.title("Wurzel des Reflexionsvermögen von parallel und senkrecht polarisierten Licht in Abhängigkeit des Einfallwinkels")

plt.savefig("O11Auswertung.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5)


plt.show()

