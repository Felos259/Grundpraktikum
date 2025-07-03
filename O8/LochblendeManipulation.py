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
# Durchmesser mit Mikroskop
B = 20 * u.ufloat(4.98, 0.02) * 10**-3
print('Durchmesser Lochblende Mikroskop: B='+ str(B) + "cm")



###############

# so viele Pixel sind 1cm
oneCm = 574.2230
# so dick ist ein cm-Strich in Pixeln
deltaOneCm = 31.0161

# Unsicherheit für 1cm ungefähr ein Viertel des Striches in beide Richtungen (also auf die Hälfte genau getroffen)
Cm = u.ufloat(1.0, (deltaOneCm/4) / oneCm )

# Fehlerbehaftung der Distance einbauen
position = RF['Distance_(unit)'] * Cm
RF['position'] = np.array([value.nominal_value for value in position])

###############
# Grey-Value Normieren

# maxGray = max(RF["Gray_Value"])
# Maximal Wert ist 255 wegen RGB - kommt aber auch beim Suchen heraus
maxGray = 255

Intensity = RF["Gray_Value"] / maxGray
RF['Intensity'] = Intensity

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


        if deltas[1]==0:
            deltas[1] = 1 * 10**.4 # vermeiden, dass Unsicherheit 0.0 beträgt
        SmoothRF.loc[k] = [means[0], np.sqrt(deltas[0]**2 + (means[0]*Cm.s)**2), means[1], deltas[1]]

#################
# Peaks bestimmen

substancialPeakDemand = int(len(SmoothRF) * 0.025)

local_max_smooth_aggr = argrelmin(SmoothRF['Intensity'].to_numpy(), order = substancialPeakDemand)
indexList = np.asarray(local_max_smooth_aggr)[0]

peaks = SmoothRF[SmoothRF.index.isin(indexList)]

peaks = peaks.sort_values('position')

# PEAKS ENTFERNEN, WEIL SIE CLUTTER SIND
peaks = peaks.iloc[3:]
peaks = peaks.iloc[:len(peaks)-2]

# Werte abspeichern
peaks.to_csv('O8/Lochblende.csv', sep=';', index = False)

#######################
# Positionen so verschieben, dass Maximum 0. Ordnung bei 0cm liegt

position = position - RF['position'].iloc[peaks.idxmax().iloc[2]*dgs]

# posminus1 = peaks.idxmax().iloc[2]
# print(posminus1)
 
halfpoint = (SmoothRF['position'].iloc[449] + SmoothRF['position'].iloc[607] )  /2


peaks['position'] = peaks['position']- halfpoint
SmoothRF['position'] = SmoothRF['position'] - halfpoint

###################################################################################################
# Plot
fig, ax = plt.subplots()

ax.set_ylim(-0.1, 1.4)
ax.set_xlim(-3.0, 3.0)

# Peaks plotten
ax.errorbar(peaks['position'],  peaks['Intensity'], 
        label = "Minima der geglätteten Daten", 
        color = 'mediumblue', linestyle='None', marker='o', markersize=8)


#Messwerte plotten
# ax.errorbar(np.array([value.n for value in position]),  Intensity, label = 'Intensität des Lichtes Lochblende', 
#             color = 'mediumblue', linestyle='None', marker='o', capsize=3, markersize=1, elinewidth = 0.5)

#Nullstellen Bessel
NS = [-19.6159, -16.471, -13.324, -10.173, -7.016, -3.832, 0.0, 3.832, 7.016, 10.173, 13.324, 16.471, 19.6159]

#Theoretische Minimapositionen
posMin = [value * lamb * SS / (np.pi * B) for value in NS]

I_0 = 1.0

def theoFunkt(x, I):
    return I * (j1( np.pi * B.n * x / (SS.n*lamb)) / (np.pi * B.n * x / (2.0*SS.n*lamb)) )**2

x_ax = np.linspace(-3, 3, 7000) 
y_ax = theoFunkt(x_ax, I_0)

ax.errorbar(x = x_ax , y = y_ax,
        label = "theoretische Funktion mit B aus dem Mikroskop", color = 'plum')

ax.errorbar(x = [value.n for value in posMin] , y = [theoFunkt(value.n, I_0) for value in posMin], xerr = [value.s for value in posMin], 
        label = "Nullstellen der theoretischen Funktion ", 
        color = 'purple', linestyle='None', marker='o', markersize=5, capsize=6)

 
# Smoothed Data plotten
ax.errorbar(x = SmoothRF['position'], y = SmoothRF['Intensity'], 
        label = "geglättete Daten - je " + str(dgs) + " Pixel zusammengefasst", 
        color = 'crimson', linestyle='None', marker='o', markersize=2, capsize=3, elinewidth = 0.5)
# xerr = SmoothRF['dPos'], yerr = SmoothRF['dInt'],

################

# cosmetics
plt.xlabel('Position $x$ in cm',fontsize=fnt)
plt.ylabel('Intensität in % des maximalen Grauwertes', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Intensitätsverteilung Lochblende", fontsize=fnt)
plt.grid()
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O8/IntensitatLochblende.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()
