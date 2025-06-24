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

RF = pd.read_csv('MessdatenO11SP.csv', header=1)
RG = pd.read_csv('MessdatenO11PP.csv', header=1)

plt.rcParams['figure.figsize'] = [19.2,10.8]

#To-Do: Unsicherheiten

RF['grundIntensitaet'] = 4.896 #U_E in V
RF['dgrundIntensitaet']=RF['grundIntensitaet']*0.0005+0.001 #Unsicherheit der Grundintensität
RG['grundIntensitaet'] = 4.896
RG['dgrundIntensitaet']=RG['grundIntensitaet']*0.0005+0.001

#Diese Korrektur ist an den Haaren herbeigezogen: Idee ist, dass bei kleinen Werten der Wert ähnlich ist.
#Die Nachnormierung (aktuell /1.125) habe ich jetzt Pi mal Daumen aus Juliens Werten.
#Wenn wir sammeln, können wir eine statistisch signifikante Auswertung der Abweichungen "guter" Messwerte bei 10° berechnen und so einen "wissenschaftlichen" Anpassungsfaktor ergaunern
RG['conradsKorrektur'] = RF['intenseSP'][1]/RG['intensePP'][1]*1.086387735

ableseWinkel = 3 #Ableseunsicherheit Winkel in Grad
sysWinkel = 0 #Systematische Unsicherheit Winkel in Grad
RF['dWinkel'] = np.sqrt(ableseWinkel**2+sysWinkel**2)
RG['dWinkel'] = np.sqrt(ableseWinkel**2+sysWinkel**2)


RF['dableseIntensitaet'] = 0.01 #Größtfehlerabschätzung der Intensität in V
RG['dableseIntensitaet'] = 0.01
RG['dsysIntensitaetPP'] = RG['intensePP']*0.0005+0.001 #Syst. Unsicherheit der Intensität in V
RF['dsysIntensitaetSP'] = RF['intenseSP']*0.0005+0.001 #Syst. Unsicherheit der Intensitt in V

RG['unsicherheitIntensitaetPP'] = np.sqrt(RG['dableseIntensitaet']**2+RG['dsysIntensitaetPP']**2)
RF['unsicherheitIntensitaetSP'] = np.sqrt(RF['dableseIntensitaet']**2+RF['dsysIntensitaetSP']**2)


# Berechnung Wurzel R für parallel polarisiertes Licht (PP) und senkrecht polarisiertes Licht (SP)
uRE = unp.uarray(RF['grundIntensitaet'],RF['dgrundIntensitaet'])
uREP = unp.uarray(RG['grundIntensitaet'],RG['dgrundIntensitaet'])
uRPP = unp.uarray((RG['intensePP']-RG['offsetIntense'])*RG['conradsKorrektur'],RG['unsicherheitIntensitaetPP'])
uRSP = unp.uarray(RF['intenseSP']-RF['offsetIntense'],RF['unsicherheitIntensitaetSP'])
print(uRPP)

uwRpp = unp.sqrt(uRPP/uREP)
uwRsp = unp.sqrt(uRSP/uRE)
RG['RPP']=np.array([value.nominal_value for value in uwRpp])
RG['dRPP']=np.array([value.s for value in uwRpp])
RF['RSP']=np.array([value.nominal_value for value in uwRsp])
RF['dRSP']=np.array([value.s for value in uwRsp])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,90/360*(2*np.pi))
ax.set_ylim(0,1.1)

#Daten
x_data = RF['winkel']/360*(2*np.pi)
x_err = RF['dWinkel']/360*(2*np.pi)

x_data_parallel = RG['winkel']/360*(2*np.pi)
x_err_parallel = RG['dWinkel']/360*(2*np.pi)

y_data_parallel = RG['RPP']
y_err_parallel = RG['dRPP']

y_data_senkrecht = RF['RSP']
y_err_senkrecht = RF['dRSP']


# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(x_data_parallel,y_data_parallel, xerr=x_err_parallel , yerr=y_err_parallel, label='Intensität des reflektierten Strahls für parallel polarisiertes Licht', color = 'darkblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2)
ax.errorbar(x_data, y_data_senkrecht, xerr=x_err , yerr=y_err_senkrecht, label='Intensität des reflektierten Strahls für senkrecht polarisiertes Licht', color = 'darkgreen', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2)

# Fitfunktion definieren
def fit_function_parallel(x, B):
    return np.sqrt(((np.tan(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.tan(x+np.arcsin(np.sin(x)/np.tan(B))))**2))

def fit_function_senkrecht(x,B):
    return np.sqrt(((np.sin(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.sin(x+np.arcsin(np.sin(x)/np.tan(B))))**2))


# Curve-Fit für parallel polarisiertes Licht
params, covariance = curve_fit(fit_function_parallel, x_data_parallel, y_data_parallel, sigma=y_err_parallel, absolute_sigma=True,bounds=[0.5,1.5])
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value_par = params[0]
B_error_par = fit_errors[0]
#Chi^2/dof berechnen
dof = len(RF.index)-len(params)
chi2 = sum([((fit_function_parallel(x,B_value_par)-y)**2)/(u**2) for x,y,u in zip(x_data_parallel,y_data_parallel,y_err_parallel)])
# Fit-Ergebnisse ausgeben
print(f"Parallel: B = {B_value_par:.6f} ± {B_error_par:.6f}")
print(f"Parallel: Chi-Quadrat/dof: {chi2/dof}")
# Fit-Ergebnisse plotten
x_ax=np.linspace(0.001, 1.55, 1000) 
y_ax_parallel = fit_function_parallel(x_ax, 0.94)
plt.plot(x_ax, y_ax_parallel, label=f"Fit zu parallel polarisierten Licht mit Fitparameter \n $\\alpha_{{B,\\text{{parallel}}}} = {B_value_par:.6f} \\pm {B_error_par:.6f}$", linewidth=2, color='blue')

# Curve-Fit für senkrecht polarisiertes Licht
params, covariance = curve_fit(fit_function_senkrecht, x_data, y_data_senkrecht, sigma=y_err_senkrecht, absolute_sigma=True, bounds=[0.5,1.5])
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value = params[0]
B_error = fit_errors[0]
#Chi^2/dof berechnen
dof = len(RF.index)-len(params)
chi2 = sum([((fit_function_senkrecht(x,B_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data_parallel,y_err_parallel)])
# Fit-Ergebnisse ausgeben
print(f"Senkrecht: B = {B_value:.6f} ± {B_error:.6f}")
print(f"Senkrecht: Chi-Quadrat/dof: {chi2/dof}")
x_ax=np.linspace(0.001, 1.55, 1000) 
y_ax_senkrecht = fit_function_senkrecht(x_ax, B_value)
plt.plot(x_ax, y_ax_senkrecht, label=f"Fit zu senkrecht polarisierten Licht mit Fitparameter \n $\\alpha_{{B,\\text{{senkrecht}}}} = {B_value:.6f} \\pm {B_error:.6f}$", linewidth=2, color='limegreen')

# Plot zeichnen
plt.xlabel('Eingangswinkel $\\alpha_E$ in Radiant',fontsize=20)
plt.ylabel('$\\sqrt{R}$ in 1',fontsize=20)
plt.legend(fontsize=20)
plt.title("Wurzel des Reflexionsvermögen von parallel und senkrecht polarisierten Licht in Abhängigkeit des Einfallwinkels",fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("O11Auswertung.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5)


plt.show()

