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

RF = pd.read_csv('Messaufgabe1.csv', header=2)

#To-Do: Unsicherheit Kapazität ergänzen
kapazitaet = unp.uarray([10**(-5)],[0]) #Kapazitaet des Kondensators in Farad

# Unsicherheiten aller Art
#ableseAmpere = 1 #Ableseunsicherheit am Amperemeter in Ampere
#sysAmpere = 0 #Systematische Unsicherheit des Amperemeters in Ampere
#RF['uI1oC']=[np.sqrt((ableseAmpere)**2+(sysAmpere)**2)]*(len(RF['I_1Weg_ohneC']))
#RF['uI1mC']=[np.sqrt((ableseAmpere)**2+(sysAmpere)**2)]*(len(RF['I_1Weg_mitC']))
#RF['uI2oC']=[np.sqrt((ableseAmpere)**2+(sysAmpere)**2)]*(len(RF['I_2Weg_ohneC']))
#RF['uI2mC']=[np.sqrt((ableseAmpere)**2+(sysAmpere)**2)]*(len(RF['I_2Weg_mitC']))

#ableseVolt2 = 1 #Ableseunsicherheit am Voltmeter 2 (nach Schaltplan) in Volt
#sysVolt2 = 0 #Systematische Unsicherheit des Voltmeters 2 (nach Schaltplan) in Volt
#RF['uV1oC']=[np.sqrt((ableseVolt2)**2+(sysVolt2)**2)]*(len(RF['U_1Weg_ohneC']))
#RF['uV1mC']=[np.sqrt((ableseVolt2)**2+(sysVolt2)**2)]*(len(RF['U_1Weg_mitC']))
#RF['uV2oC']=[np.sqrt((ableseVolt2)**2+(sysVolt2)**2)]*(len(RF['U_2Weg_ohneC']))
#RF['uV2mC']=[np.sqrt((ableseVolt2)**2+(sysVolt2)**2)]*(len(RF['U_2Weg_mitC']))

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,0.11)
ax.set_ylim(0,1000)

#Daten
x_data_1 = RF['I_1Weg_ohneC']*10**(-3)
x_err_1 = RF['uI1oC']
y_data_1 = RF['U_1Weg_ohneC']*10/3
y_err_1 = RF['uV1oC']

x_data_2 = RF['I_1Weg_mitC']*10**(-3)
x_err_2 = RF['uI1mC']
y_data_2 = RF['U_1Weg_mitC']*10/3
y_err_2 = RF['uV1mC']

x_data_3 = RF['I_2Weg_ohneC']*10**(-3)
x_err_3 = RF['uI2oC']
y_data_3 = RF['U_2Weg_ohneC']*10/3
y_err_3 = RF['uV2oC']

x_data_4 = RF['I_2Weg_mitC']*10**(-3)
x_err_4 = RF['uI2mC']
y_data_4 = RF['U_2Weg_mitC']*10/3
y_err_4 = RF['uV2mC']

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(x_data_1,y_data_1, xerr=x_err_1 , yerr=y_err_1, label='U(I) für E1', color = 'lightblue', linestyle='None', marker='o', capsize=6)
ax.errorbar(x_data_2, y_data_2, xerr=x_err_2 , yerr=y_err_2, label='U(I) für E2', color = 'darkblue', linestyle='None', marker='o', capsize=6)
ax.errorbar(x_data_3, y_data_3, xerr=x_err_3 , yerr=y_err_3, label='U(I) für Z1', color = 'lightgreen', linestyle='None', marker='o', capsize=6)
ax.errorbar(x_data_4, y_data_4, xerr=x_err_4 , yerr=y_err_4, label='U(I) für Z2', color = 'darkgreen', linestyle='None', marker='o', capsize=6)

# Fitfunktion definieren
def fit_function(x, A, x0):
    return A * (x - x0)


# Curve-Fit mit für EINWEG ohne Kondensator
params_1, covariance_1 = curve_fit(fit_function, x_data_1, y_data_1, sigma=y_err_1, absolute_sigma=True)
fit_errors_1 = np.sqrt(np.diag(covariance_1))  # Fehler der Fit-Parameter
A_1_value = params_1[0]
A_1_error = fit_errors_1[0]
x0_1_value = params_1[1]
x0_1_error = fit_errors_1[1]
#Chi^2/dof berechnen
dof_1 = len(RF.index)-len(params_1)
chi2_1 = sum([((fit_function(x,A_1_value,x0_1_value)-y)**2)/(u**2) for x,y,u in zip(x_data_1,y_data_1,y_err_1)])
# Fit-Ergebnisse ausgeben
print(f"1-Weg ohne Kondensator: A = {A_1_value:.6f} ± {A_1_error:.6f} (Innenwiderstand EWG)")
print(f"1-Weg ohne Kondensator: x0 = {x0_1_value:.6f} ± {x0_1_error:.6f} (Kurzschlussstrom EWG)")
U0_1=-unp.uarray([x0_1_value],[x0_1_error])*(unp.uarray([A_1_value],[A_1_error]))
print("Leerlaufspannung EWG: ", U0_1)
print(f"1-Weg ohne Kondensator: Chi-Quadrat/dof: {chi2_1/dof_1}")
# Fit-Ergebnisse plotten
x_ax_1=np.linspace(0, 300, 1000) 
y_ax_1 = fit_function(x_ax_1, A_1_value,x0_1_value)
plt.plot(x_ax_1, y_ax_1, label=f"E1-Fit: $y = A \\cdot (x-x_0)$ \n $A = {A_1_value:.6f} \\pm {A_1_error:.6f}$ \n $x_0 = {x0_1_value:.6f} \\pm {x0_1_error:.6f}$", linewidth=2, color='blue')

# Curve-Fit mit für ZWEIWEG ohne Kondensator
params_3, covariance_3 = curve_fit(fit_function, x_data_3, y_data_3, sigma=y_err_3, absolute_sigma=True)
fit_errors_3 = np.sqrt(np.diag(covariance_3))  # Fehler der Fit-Parameter
A_3_value = params_3[0]
A_3_error = fit_errors_3[0]
x0_3_value = params_3[1]
x0_3_error = fit_errors_3[1]
#Chi^2/dof berechnen
dof_3 = len(RF.index)-len(params_3)
chi2_3 = sum([((fit_function(x,A_3_value,x0_3_error)-y)**2)/(u**2) for x,y,u in zip(x_data_3,y_data_3,y_err_3)])
# Fit-Ergebnisse ausgeben
print(f"2-Weg ohne Kondensator: A = {A_3_value:.6f} ± {A_3_error:.6f} (Innenwiderstand ZWG)")
print(f"2-Weg ohne Kondensator: x0 = {x0_3_value:.6f} ± {x0_3_error:.6f} (Kurzschlussstrom ZWG)")
U0_3=-unp.uarray([x0_3_value],[x0_3_error])*(unp.uarray([A_3_value],[A_3_error]))
print("Leerlaufspannung EWG: ", U0_3)
print(f"2-Weg ohne Kondensator: Chi-Quadrat/dof: {chi2_3/dof_3}")
x_ax_3=np.linspace(0, 300, 1000) 
y_ax_3 = fit_function(x_ax_3, A_3_value,x0_3_value)
plt.plot(x_ax_3, y_ax_3, label=f"Z1-Fit: $y = A \\cdot (x-x_0)$ \n $A = {A_3_value:.6f} \\pm {A_3_error:.6f}$ \n $x_0 = {x0_3_value:.6f} \\pm {x0_3_error:.6f}$", linewidth=2, color='limegreen')

# Plot zeichnen
plt.xlabel('Laststrom $I$ in A')
plt.ylabel("Zeitlicher Mittelwert der Gleichspannungen $U$ in V")
plt.legend()
plt.title("Zeitliche Mittelwerte der Gleichspannungen in Abhängigkeit des Laststroms")

plt.savefig("Messreihe1.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5)


# Residuenplot E1 :)
residuen_1 = (unp.uarray(y_data_1,y_err_1) - u.ufloat(A_1_value,A_1_error)*(unp.uarray(x_data_1,x_err_1)-u.ufloat(x0_1_value,x0_1_error)))

residuen_1_value = np.array([value.nominal_value for value in residuen_1])
residuen_1_error = np.array([value.s for value in residuen_1])

fig, ax = plt.subplots()
ax.set_xlim(-0.001,0.05)
ax.set_ylim(-50,20)

plt.errorbar(x_data_1, residuen_1_value, xerr=x_err_1, yerr=residuen_1_error, fmt='o', label='Residuen', capsize=5, color='red')
plt.tick_params(axis="both",direction="in",top=True,left=True,right=True,bottom=True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontale Linie bei y=0
plt.xlabel('Laststrom $I$ in A')
plt.ylabel('Residuen $(U_i - \\hat{U}_i)$ in V')
plt.legend()
plt.title("Residuendarstellung zum $U$-$I$-Diagramm für den Aufbau E1")

plt.savefig("ResiduumE1.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)


# Residuenplot Z1 :)
residuen_3 = (unp.uarray(y_data_3,y_err_3) - u.ufloat(A_3_value,A_3_error)*(unp.uarray(x_data_3,x_err_3)-u.ufloat(x0_3_value,x0_3_error)))

residuen_3_value = np.array([value.nominal_value for value in residuen_3])
residuen_3_error = np.array([value.s for value in residuen_3])

fig, ax = plt.subplots()
ax.set_xlim(-0.001,0.11)
ax.set_ylim(-26,22)

plt.errorbar(x_data_3, residuen_3_value, xerr=x_err_3, yerr=residuen_3_error, fmt='o', label='Residuen', capsize=5, color='red')
plt.tick_params(axis="both",direction="in",top=True,left=True,right=True,bottom=True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontale Linie bei y=0
plt.xlabel('Laststrom $I$ in A')
plt.ylabel('Residuen $(U_i - \\hat{U}_i)$ in V')
plt.legend()
plt.title("Residuendarstellung zum $U$-$I$-Diagramm für den Aufbau Z1")

plt.savefig("ResiduumZ1.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 


plt.show()

