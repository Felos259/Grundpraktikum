import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.special import j1
import iminuit as i
import uncertainties as u
import uncertainties.umath as um

from uncertainties import unumpy as unp

RF = pd.read_csv('O8/PlotProfile_Lochblende.csv', header=0, sep=',')

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

# degree of smoothness - Anzahl der Bits, die zusammengefasst werden
dgs = 4
lamb = 532 * 10**-6
SS = u.ufloat(86.5, 0.5) - u.ufloat(2.0, 0.5) 
# Durchmesser mit Mikroskop in Centimeter
Blende = 20 * u.ufloat(4.98, 0.02) * 10**-4

print('Durchmesser Lochblende Mikroskop: B='+ str(Blende.n) + "cm")

# Unsicherheitsfaktor für Intensität
uInt = 0.1

###############

# so viele Pixel sind 1cm
oneCm = 574.2230
# so dick ist ein cm-Strich in Pixeln
deltaOneCm = 31.0161

# Unsicherheit für 1cm ungefähr ein Viertel des Striches in beide Richtungen (also auf die Hälfte genau getroffen)
Cm = u.ufloat(1.0, (deltaOneCm/4) / oneCm )

# Rauschen rausschmeißen
RF =  RF[(RF['Distance_(unit)']  >= 1.4)& (RF['Distance_(unit)']  <= 6)]

# Fehlerbehaftung der Distance einbauen

position = RF['Distance_(unit)'] * Cm
RF['position'] = np.array([value.nominal_value for value in position])

###############
# Grey-Value Normieren

# maxGray = max(RF["Gray_Value"])
# Maximal Wert ist 255 wegen RGB - kommt aber auch beim Suchen heraus
maxGray = 255

RF['Intensity'] = RF["Gray_Value"] / maxGray
RF['dInt'] = uInt * RF['Intensity']



##################
# Smooth data

length = len(RF.Intensity)
rest = length

SmoothRF = pd.DataFrame(columns=['position', 'dPos', 'Intensity', 'dInt'])

for k in range(0, int(length/dgs) , 1):
    # Index des ersten Pixel
    j = dgs * k + 1
    # Wenn noch genug Pixel übrig sind, um sie zu vereinigen
    if ((rest - j) >= dgs):
        col = ['Distance_(unit)', 'Intensity']
        means = [0.0, 0.0]
        deltas = [0.0, 0.0]

        for p in range(0, 2, 1):
            # Zeilen der Spalte col[p] in RF - Index j bis j+dgs
            array = RF[col[p]][j:j+dgs]
            #print(array)

            # Mittelwert über die dgs vielen Pixel
            means[p] = np.mean(array)
            #print('MEAN: ', means[p])

            # Standardabweichung über die dgs vielen Pixel 
            std = np.std(array, ddof=1)
            #print('STD: ', std)

            # statistische Unsicherheit der dgs vielen Pixel
            deltas[p] = std * np.sqrt(dgs)
            #print('DELTA STD: ', deltas[p])

        if deltas[1]==0 :
            deltas[1] = 1.0 * 10**-4 # vermeiden, dass Unsicherheit 0.0 beträgt
        SmoothRF.loc[k] = [means[0], np.sqrt(deltas[0]**2 + (means[0]*Cm.s)**2), means[1], np.sqrt(deltas[1]**2 + (means[1]*uInt)**2)]
       
    

#################
# Peaks bestimmen

substancialPeakDemand = int(len(SmoothRF) * 0.025)

local_max_smooth_aggr = argrelmin(SmoothRF['Intensity'].to_numpy(), order = substancialPeakDemand)
indexList = np.asarray(local_max_smooth_aggr)[0]

peaks = SmoothRF[SmoothRF.index.isin(indexList)]

peaks = peaks.sort_values('position')

# PEAKS ENTFERNEN, WEIL SIE CLUTTER SIND
# peaks = peaks.iloc[3:]
# peaks = peaks.iloc[:len(peaks)-2]

# Werte abspeichern
peaks.to_csv('O8/Lochblende.csv', sep=';', index = False)

#######################
# Positionen so verschieben, dass Maximum 0. Ordnung bei 0cm liegt

halfpoint = (SmoothRF['position'].iloc[248] + SmoothRF['position'].iloc[406] )  /2

position = position - halfpoint
peaks['position'] = peaks['position'] - halfpoint
SmoothRF['position'] = SmoothRF['position'] - halfpoint

SmoothRF.to_csv('O8/SmoothCentral.csv', sep=';', index = False)

###################################################################################################
# Plot
fig, ax = plt.subplots()

ax.set_ylim(-0.1, 1.6)
# ax.set_xlim(-4.0, 4.0)

# Peaks plotten
ax.errorbar(peaks['position'],  peaks['Intensity'], xerr= peaks['dPos'], yerr=peaks['dInt'],
        label = "Minima der geglätteten Daten", 
        color = 'lightgreen', linestyle='None', marker='o', markersize=8, capsize = 10)


#Nullstellen Bessel
NS = [ -13.324, -10.173, -7.016, -3.832, 0.0, 3.832, 7.016, 10.173, 13.324] # , 16.471, 19.6159

x_data = SmoothRF['position'].iloc[indexList[3]:indexList[len(indexList)-1]] 
y_data = SmoothRF['Intensity'].iloc[indexList[3]:indexList[len(indexList)-1]] 
y_err  = SmoothRF['dInt'].iloc[indexList[3]:indexList[len(indexList)-1]] 


def fit_function(x, I, B, A):
    return I * ( j1( (np.pi * B * x )/(SS.n*lamb) ) / ( (np.pi * B * x)/(2.0*SS.n*lamb) ) )**2 + A

# ax.errorbar(x = x_ax , y = y_ax,
#         label = f"theoretische Funktion mit \n $B$={b}cm und $I_0$={I_0}", color = 'plum')

# Curve-Fit mit Unsicherheiten in y                                                                      
params, covariance = curve_fit(fit_function, x_data, y_data, sigma=y_err, absolute_sigma=True, bounds = ([ -np.inf , 0.0 , -np.inf ], [ np.inf, 1.0 , np.inf ]), p0 = [1.0, 0.0, 0.0] )
I_value = params[0]
B_value = params[1]
A_value = params[2]
fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
I_error = fit_errors[0]
B_error = fit_errors[1]
A_error = fit_errors[2]

posPeaks = unp.uarray(peaks['position'], peaks['dPos'])


#Brennweite
f = 7.8 # cm
print(1.22 *2.0 * f * posPeaks / SS)
# u.ufloat(B_value, B_error)

dof = len(RF.index)-len(params)
chi2 = sum([((fit_function(x,I_value, B_value, A_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# Fit-Ergebnisse ausgeben
#print(f"A = {A_value:.6f} ± {A_error:.6f}")
#print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
#print(f"Chi-Quadrat/dof: {chi2/dof}")

#Theoretische Minimapositionen
posMin = [value * lamb * SS / (np.pi * B_value) for value in NS]

x_ax = np.linspace(posMin[0].n+0.05, posMin[len(posMin)-1].n-0.05, 1000) 
y_ax = fit_function(x_ax, I_value, B_value, A_value)

label = "Fit nach $y = I_0 \\cdot \\left(  \\frac{  J_1(\\frac{\\pi \\cdot B \\cdot \sin \\alpha}{\\lambda})   }{ \\frac{\\pi \cdot B \\cdot \sin \\alpha }{2 \\lambda}   }  \\right)^2 + A$: "
label += f"\n $I_0={I_value:.6f}\pm {I_error:.6f}$ \n $B={B_value:.6f}\pm {B_error:.6f}$cm \n $A={A_value:.6f}\pm {A_error:.6f}$"

# Plot zeichnen
plt.plot(x_ax, y_ax, label = label, linewidth = 2, color = 'plum')

ax.set_xlim(posMin[0].n+0.05, posMin[len(posMin)-1].n-0.05)


##########################

ax.errorbar(x = [value.n for value in posMin] , y = [fit_function(value.n, I_value, B_value, A_value) for value in posMin], xerr = [value.s for value in posMin], 
        label = "Nullstellen der theoretischen Funktion ", 
        color = 'purple', linestyle='None', marker='o', markersize=5, capsize=6)

 
# Smoothed Data plotten
ax.errorbar(x = SmoothRF['position'], y = SmoothRF['Intensity'],
        label = "geglättete Daten - $dgs$=" + str(dgs) , 
        color = 'crimson', linestyle='None', marker='o', markersize=3, capsize=3, elinewidth = 0.5) # yerr  = SmoothRF['dInt'],


# Messwerte plotten
ax.errorbar(x= np.array([value.n for value in position]) , y=RF['Intensity'], label = 'Intensität des Lichtes Lochblende', 
            color = 'mediumblue', linestyle='None', marker='o', capsize=3, markersize=1, elinewidth = 0.5)

################

# cosmetics
plt.xlabel('Position $x$ in cm',fontsize=fnt)
plt.ylabel('relative Intensität', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Intensitätsverteilung Lochblende", fontsize=fnt)
plt.grid()
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O8/IntensitatLochblende.pdf", format='pdf', bbox_inches='tight') 
plt.show()
