import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
import iminuit as i
import uncertainties as u
import uncertainties.umath as um

from uncertainties import unumpy as unp

RF = pd.read_csv('O8/PlotProfile_Gitter.csv', header=0, sep=',')

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

# degree of smoothness - Anzahl der Bits, die zusammengefasst werden
dgs = 4

###############

# so viele Pixel sind 1cm
oneCm = 640.0383
# so dick ist ein cm-Strich in Pixeln
deltaOneCm = 43

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


        # Index 0 => Mittelwert über dgs Pixel bzgl Position   
        # Index 1 => Unsicherheit Index 0 
        # Unsicherheit ergibt sich aus statistischer Unsicherheit und systematischer Unsicherheit (Wert * Cm)

        # Index 2 => Mittelwert über dgs Pixel bzgl Intensity 
        # Index 3 => Unsicherheit Index 2
        # Unsicherheit ist rein Statistisch, da wir nichts über die Genauigkeit 
        SmoothRF.loc[k] = [means[0], np.sqrt(deltas[0]**2 + (means[0]*Cm.s)**2), means[1], deltas[1]]

#################
# Peaks bestimmen

substancialPeakDemand = int(len(SmoothRF) * 0.03)

local_max_smooth_aggr = argrelmax(SmoothRF['Intensity'].to_numpy(), order = substancialPeakDemand)
indexList = np.asarray(local_max_smooth_aggr)[0]

peaks = SmoothRF[SmoothRF.index.isin(indexList)]

peaks = peaks.sort_values('position')

# ERSTE ZWEI UND LETZTEN PEAK ENTFERNEN, WEIL SIE SHUTTER SIND
peaks = peaks.iloc[2:]
peaks = peaks.iloc[:len(peaks)-1]

# Werte abspeichern
peaks.to_csv('O8/Gitter.csv', sep=';', index = False)


#################
# Plot

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 1.4)


# Peaks plotten
plt.plot(peaks['position'],  peaks['Intensity'], 
        label = "Maxima der geglätteten Daten", 
        color = 'lightgreen', linestyle='None', marker='o', markersize=8)
 
# Smoothed Data plotten
ax.errorbar(x = SmoothRF['position'], y = SmoothRF['Intensity'], 
        label = "geglättete Daten - je " + str(dgs) + " Pixel zusammengefasst", 
        color = 'crimson', linestyle='None', marker='o', markersize=3, capsize=3, elinewidth = 0.5)
# xerr = SmoothRF['dPos'], yerr = SmoothRF['dInt'],

#Messwerte plotten
#Daten
x_data = RF['position']
x_err = np.array([value.s for value in position])
y_data = Intensity


ax.errorbar(x_data, y_data, label = 'Intensität des Lichtes Gitter', 
            color = 'mediumblue', linestyle='None', marker='o', markersize=1, elinewidth = 0.5 )
# xerr = x_err,


################

# cosmetics

plt.xlabel('Position $x$ in cm',fontsize=fnt)
plt.ylabel('Intensität in % des maximalen Grauwertes', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Intensitätsverteilung Gitter", fontsize=fnt)
plt.grid()
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O8/IntensitatGitter.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()
