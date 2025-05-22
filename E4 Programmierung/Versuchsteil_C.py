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


RF = pd.read_csv('Versuchsteil_C.csv', header=1)
RG = pd.read_csv('Versuchsteil_C_phi.csv', header=1)

uf = unp.uarray(RF['Frequenz'], RF['df'])
uuch1 = unp.uarray(RF['U_Ch1'], RF['dU_Ch1'])
uuch2 = unp.uarray(RF['U_Ch2'], RF['dU_Ch2'])
uphi = unp.uarray(RF['phi'],RF['dphi'])

############################### Kapitel 8.3.1 #############################################################

# R_RL = Summe aus Omhschen Widerstand und Spulenwiderstand
widerstand = unp.uarray([10.16],[0.2016])+unp.uarray([151.02],[10*0.01+1.5102]) #in Ohm
wid=161.18/1000 #in Kilo-Ohm
uwid=1.6227712716214817/1000 #in Kilo-Ohm

# Scheinwiderstand bestimmen
uscheinwiderstand = widerstand*(uuch1/uuch2)/1000 #in kOhm 
RF['Z_RLC']=np.array([value.nominal_value for value in uscheinwiderstand])
RF['dZ_RLC']=np.array([value.s for value in uscheinwiderstand])

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,1600)
ax.set_ylim(0, 90)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RF.Frequenz, RF.Z_RLC, xerr=RF.df , yerr=RF.dZ_RLC, label='$|Z_{RLC}|$ in Abhängigkeit der Frequanz', color = 'lightblue', linestyle='None', marker='o', capsize=6)

# linearer Fit

# Fitfunktion definieren
# def fit_function(x, A, B): #A = L und B = C laut Theorie
#     return np.sqrt((wid)**2+((2*np.pi*A*x)-(1/(2*np.pi*x * B)))**2)
#A entspricht der Induktivität der Spule L, x0 entspricht R_RL, B entspricht der Kapazität C

def fit_function(x, A, B): #A = L und B = C laut Theorie
    return np.sqrt((wid)**2+(2*np.pi*A*x)**2-2(A/B)+(1/(2*np.pi*x * B)**2))

#Daten
x_data = RF['Frequenz']
x_err = RF['df']
y_data = RF['Z_RLC'] #in Kilo-Ohm
y_err = RF['dZ_RLC']

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True,bounds = ([0,0],[50,10**(-3)]))
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
B_value = params[1]
B_error = fit_errors[1]
#x0_value = params[2]
#x0_error = fit_errors[2]

dof = len(RF.index)-len(params)
chi2 = sum([(fit_function(x,A_value,B_value)-y)**2/u**2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
print(f"B = {B_value:.9f} ± {B_error:.9f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

# Minimum berechnen
induktivitaet = unp.uarray([A_value],[A_error])*1000 #in H
i_value = A_value #in kH
print("Induktivität (H): ", induktivitaet)
kapazitaet = unp.uarray([B_value],[B_error])/1000 #in F
k_value = B_value #in mF
print("Kapazität (F): ", kapazitaet)
#ohmscherWiderstand = unp.uarray([x0_value],[x0_error])*1000
#w_value = x0_value*1000
#print("Widerstand (Ohm):", ohmscherWiderstand)
minimum = 1/(2*np.pi*unp.sqrt(induktivitaet*kapazitaet))
print("Das Minimum liegt bei f=", minimum," Hz")


x_ax=np.linspace(0.0001, 1600, 10000) #Hier NICHT bei 0 anfangen, da man sonst einen Divide-by-zero-Fehler erzeugt!
y_ax = fit_function(x_ax, A_value,B_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = \\sqrt{{((A \\cdot 2\\cdot \\pi \\cdot x- \\frac{{1}}{{(B\\cdot 2\\cdot \\pi\\cdot x)}})^2+x_0^2)}}$ \n $A = {A_value:.9f} \\pm {A_error:.9f}$ \n $B = {B_value:.9f} \\pm {B_error:.9f}$ \n $x_0= {wid:.6f} \\pm {uwid:.6f}$", linewidth=2, color='blue')
plt.xlabel('Frequenz $f$ in $s^{-1}$')
plt.ylabel("$|Z_{LRC}|$ in $k\\Omega$")
plt.legend()
plt.title("$f$-$|Z_{LRC}|$-Diagramm")

plt.savefig("VersuchsteilC1.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("VersuchsteilC1.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 
#plt.show()


# 2D list of variables (tabular data with rows and columns)
input_variable = [["Frequenz","df","Z_LRC","dZ_LRC"],[x_data,x_err,y_data,y_err]]
 
# Example.csv gets created in the current working directory
with open ('WertetabelleVersuchsteilCTeil1.csv','w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerows(input_variable)


############################### Kapitel 8.3.2 #############################################################

# Figure und Subplots erstellen - bei denen alle Subplots die gleichen Achsen haben
fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0,1600)
ax.set_ylim(-np.pi, 3*np.pi)

# Plot der Messwerte V und p mit Errorbars 
ax.errorbar(RG.Frequenz, RG.phi_besser, xerr=RG.df , yerr=RG.dphi, label='$tan(\\varphi)$ in Abhängigkeit der Frequanz', color = 'lightblue', linestyle='None', marker='o', capsize=6)


# Fitfunktion definieren
def fit_function2(x, A, B): #A = L und B = C laut Theorie
    return ((2*np.pi*x*A)-1/(2*np.pi*x*B))/(wid*1000)
#A entspricht der Induktivität der Spule L, x0 entspricht R_RL, B entspricht der Kapazität C

#Daten
x_data = RG['Frequenz']
x_err = RG['df']
y_data = RG['phi_besser']
y_err = RG['dphi']

# Curve-Fit mit Unsicherheiten in y
params, covariance = curve_fit(fit_function2, x_data, y_data, sigma=y_err, absolute_sigma=True)#, bounds = ([0,0],[50,10**(-3)]))
A_value = params[0]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
A_error = fit_errors[0]
B_value = params[1]
B_error = fit_errors[1]
#x0_value = params[2]
#x0_error = fit_errors[2]

dof = len(RG.index)-len(params)
chi2 = sum([(fit_function2(x,A_value,B_value)-y)**2/u**2 for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
print(f"A = {A_value:.6f} ± {A_error:.6f}")
print(f"B = {B_value:.6f} ± {B_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
print(f"Chi-Quadrat/dof: {chi2/dof}")

x_ax=np.linspace(0.0001, 1600, 10000) #Hier NICHT bei 0 anfangen, da man sonst einen Divide-by-zero-Fehler erzeugt!
y_ax = fit_function2(x_ax, A_value,B_value)

# Plot zeichnen
plt.plot(x_ax, y_ax, label=f"Fit: $y = \\frac{{2\\cdot\\pi\\cdot x\\cdot A-\\frac{{1}}{{2\\cdot\\pi\\cdot x\\cdot B}}}}{{x_0}}$ \n $A = {A_value:.6f} \\pm {A_error:.6f}$ \n $B = {B_value:.6f} \\pm {B_error:.6f}$ \n $x_0= {wid*1000:.6f} \\pm {uwid*1000:.6f}$", linewidth=2, color='blue')
plt.plot(x_ax, y_ax, linewidth=2, color='blue')
#k_alt_value=107.4*10**(-9)
#i_alt_value=9.22

# In der folgenden Zeile ist das Problem... Einheiten: Widerstand in Kilo-Ohm, Induktivität (I-Value) in kH, Kapazität (k-Value) in mF 
plt.plot(x_ax, (2*np.pi/(wid)*x_ax*i_value-1/(2*np.pi*wid*x_ax*k_value)),label=f"Aus den Werten von $C$,$L$, $R_{{RL}}$ resultierender Fit", linewidth=1.5, color='orange')
#Teilt man die Zeile über dieser Zeile durch 1000, so kommt ein relativ okayer Fit heraus
plt.xlabel('Frequenz $f$ in $s^{-1}$')
plt.ylabel("Tangens der Phasenverschiebung $\\tan(\\varphi)$")
plt.legend()
plt.title("$f$-$Tan(\\varphi)$-Diagramm")

plt.savefig("VersuchsteilC2mitFits.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
#plt.savefig("VersuchsteilC2.svg", format='svg', bbox_inches='tight', pad_inches=0.5) 

plt.show()

