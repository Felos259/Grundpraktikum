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

SP = pd.read_csv('MessdatenO11SP.csv', header=1)
PP = pd.read_csv('MessdatenO11PP.csv', header=1)
BN = pd.read_csv('BerechnungNormierung.csv',)

plt.rcParams['figure.figsize'] = [19.2,10.8]

#Senkrect polarisiertes Lict
SP['wU_E'] =4.896
SP['dU_E']=SP['wU_E']*0.0005+0.001
SP['uU_E']=unp.uarray(SP['wU_E'],SP['dU_E'])

SP['dWinkel']=3 #Ableseunsicherheit in Grad
SP['uWinkel']=unp.uarray(SP['winkel'],SP['dWinkel'])
SP['uWinkel_rad']=SP['uWinkel']/360*(2*np.pi)
SP['winkel_rad']=np.array([value.nominal_value for value in SP['uWinkel_rad']])
SP['dWinkel_rad']=np.array([value.s for value in SP['uWinkel_rad']])

SP['wIntensitaet']=SP['intenseSP']-SP['offsetIntense']
SP['dIntensitaet_ablese']=0.01
SP['dIntensitaet_sys']=SP['intenseSP']*0.0005+0.001
SP['dIntensitaet']=np.sqrt(SP['dIntensitaet_ablese']**2+SP['dIntensitaet_sys']**2)
SP['uIntensitaet']=unp.uarray(SP['wIntensitaet'],SP['dIntensitaet'])

SP['uR']=SP['uIntensitaet']/(SP['uU_E'])
SP['uWurzelR']=unp.sqrt(SP['uR'])
SP['RSP']=np.array([value.nominal_value for value in SP['uWurzelR']])
SP['dRSP']=np.array([value.s for value in SP['uWurzelR']])


#Parellel polarisiertes Lict
PP['dWinkel']=3 #Ableseunsichereit in Grad
PP['uWinkel']=unp.uarray(PP['winkel'],PP['dWinkel'])
PP['uWinkel_rad']=PP['uWinkel']/360*(2*np.pi)
PP['winkel_rad']=np.array([value.nominal_value for value in PP['uWinkel_rad']])
PP['dWinkel_rad']=np.array([value.s for value in PP['uWinkel_rad']])

PP['wIntensitaet']=PP['intensePP']-PP['offsetIntense']
PP['dIntensitaet_ablese']=0.01
PP['dIntensitaet_sys']=PP['intensePP']*0.0005+0.001
PP['dIntensitaet']=np.sqrt(PP['dIntensitaet_ablese']**2+PP['dIntensitaet_sys']**2)
PP['uIntensitaet']=unp.uarray(PP['wIntensitaet'],PP['dIntensitaet'])


#######################################################################################
#Korrektur

## Lade Werte von meinen Freunden
BN['uR_SP']=unp.uarray(BN['R_SP'],BN['dR_SP'])
BN['uR_PP']=unp.uarray(BN['R_PP'],BN['dR_PP'])
BN['q']=BN['uR_SP']/BN['uR_PP']
BN['q_Wert']=np.array([value.nominal_value for value in BN['q']])

## Berechne Mittelwert und Unsicherheit
mean = np.mean(BN['q_Wert'])
print("Mean: ", mean)
std = np.std(BN['q_Wert'], ddof=1)
print("Std: ", std)
deltaStd = std/np.sqrt(6)
print("Unsicherheit: ", deltaStd)

## Siehe LaTeX-Bericht für Rechenweg
qMean=u.ufloat(mean,deltaStd)
r_SP_15Grad_normiert = SP['uR'][1]
r_PP_15Grad_korr = qMean*SP['uR'][1]
print("r_PP_15Grad_korr:", r_PP_15Grad_korr)
r_PP_15Grad_mess = PP['uIntensitaet'][1]
beta = r_PP_15Grad_korr/r_PP_15Grad_mess ## Dies ist (hoffentlich) der Korrekturfaktor
print("Beta:", beta)
######################################################################################

PP['uR']=PP['uIntensitaet']*beta
PP['uWurzelR']=unp.sqrt(PP['uR'])
PP['RPP']=np.array([value.nominal_value for value in PP['uWurzelR']])
PP['dRPP']=np.array([value.s for value in PP['uWurzelR']])


# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,90/360*(2*np.pi))
ax.set_ylim(0,1)

#Daten
x_data_SP = SP['winkel_rad']
x_err_SP = SP['dWinkel_rad']

x_data_PP = PP['winkel_rad']
x_err_PP = PP['dWinkel_rad']

y_data_SP = SP['RSP']
y_err_SP = SP['dRSP']

y_data_PP = PP['RPP']
y_err_PP = PP['dRPP']


# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(x_data_PP,y_data_PP, xerr=x_err_PP, yerr=y_err_PP, label='Intensität des reflektierten Strahls für parallel polarisiertes Licht', color = 'darkblue', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2)
ax.errorbar(x_data_SP, y_data_SP, xerr=x_err_SP , yerr=y_err_SP, label='Intensität des reflektierten Strahls für senkrecht polarisiertes Licht', color = 'darkgreen', linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2)

# Fitfunktion definieren
def fit_function_PP(x, B):
    return np.sqrt(((np.tan(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.tan(x+np.arcsin(np.sin(x)/np.tan(B))))**2))

def fit_function_SP(x,B):
    return np.sqrt(((np.sin(x-np.arcsin(np.sin(x)/np.tan(B))))**2)/((np.sin(x+np.arcsin(np.sin(x)/np.tan(B))))**2))


# Curve-Fit für parallel polarisiertes Licht
params, covariance = curve_fit(fit_function_PP, x_data_PP, y_data_PP, sigma=y_err_PP, absolute_sigma=True,bounds=[0.5,1.5])
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value_PP = params[0]
B_error_PP = fit_errors[0]
#Chi^2/dof berechnen
dof = len(PP.index)-len(params)
chi2 = sum([((fit_function_PP(x,B_value_PP)-y)**2)/(u**2) for x,y,u in zip(x_data_PP,y_data_PP,y_err_PP)])
# Fit-Ergebnisse ausgeben
print(f"Parallel: B = {B_value_PP:.6f} ± {B_error_PP:.6f}")
print(f"Parallel: Chi-Quadrat/dof: {chi2/dof}")
# Fit-Ergebnisse plotten
x_ax=np.linspace(0.001, 2, 1000) 
y_ax_PP = fit_function_PP(x_ax, 0.94)
plt.plot(x_ax, y_ax_PP, label=f"Fit zu parallel polarisierten Licht mit Fitparameter \n $\\alpha_{{B,\\text{{parallel}}}} = {B_value_PP:.6f} \\pm {B_error_PP:.6f}$", linewidth=2, color='blue')

# Curve-Fit für senkrecht polarisiertes Licht
params, covariance = curve_fit(fit_function_SP, x_data_SP, y_data_SP, sigma=y_err_SP, absolute_sigma=True, bounds=[0.5,1.5])
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
B_value_SP = params[0]
B_error_SP= fit_errors[0]
#Chi^2/dof berechnen
dof = len(SP.index)-len(params)
chi2 = sum([((fit_function_SP(x,B_value_SP)-y)**2)/(u**2) for x,y,u in zip(x_data_SP,y_data_SP,y_err_SP)])
# Fit-Ergebnisse ausgeben
print(f"Senkrecht: B = {B_value_SP:.6f} ± {B_error_SP:.6f}")
print(f"Senkrecht: Chi-Quadrat/dof: {chi2/dof}")
x_ax=np.linspace(0.001, 2, 1000) 
y_ax_SP = fit_function_SP(x_ax, B_value_SP)
plt.plot(x_ax, y_ax_SP, label=f"Fit zu senkrecht polarisierten Licht mit Fitparameter \n $\\alpha_{{B,\\text{{senkrecht}}}} = {B_value_SP:.6f} \\pm {B_error_SP:.6f}$", linewidth=2, color='limegreen')

# Plot zeichnen
plt.xlabel('Eingangswinkel $\\alpha_E$ in Radiant',fontsize=20)
plt.ylabel('$\\sqrt{R}$ in 1',fontsize=20)
plt.legend(fontsize=20)
plt.title("Wurzel des Reflexionsvermögen von parallel und senkrecht polarisierten Licht in Abhängigkeit des Einfallwinkels",fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("O11Auswertung.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5)


plt.show()

