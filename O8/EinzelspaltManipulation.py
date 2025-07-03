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
from scipy.signal import argrelmin
from scipy.signal import find_peaks

fnt = 20 # fontsize for zooming, default 10
plt.rcParams['figure.figsize'] = [19.2,10.8]

# degree of smoothness - Anzahl der Bits, die zusammengefasst werden
dgs = 4

RF = pd.read_csv('O8/PlotProfile_Einzelspalt.csv', header=0, sep=',')

###############

# so viele Pixel sind 1cm
oneCm = 635.0504
# so dick ist ein cm-Strich in Pixeln
deltaOneCm = 28

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

# print(SmoothRF)

#################
# Plot der Intensität und Peaks bestimmen


fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke

# Achsen richten
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 1.4)


#Daten
x_data = RF['position']
x_err = np.array([value.s for value in position])
y_data = Intensity

######################

# lokale Minima und Maxima bestimmen und anzeigen lassen

# number of points to be checked before and after

# local_min_vals = RF.loc[ RF['Intensity'] == RF['Intensity'].rolling(n, center=True).min() ]
# local_max_vals = RF.loc[ RF['Intensity'] == RF['Intensity'].rolling(n, center=True).max() ]

# # Plot results
# plt.scatter(x=local_min_vals.position, y= local_min_vals.Intensity, c='r')
# plt.scatter(local_max_vals.position, local_max_vals.Intensity, c='g')

# local_min = alle Indizes in RF['Inentisty'], die Peaks sind
# prop = ein paar weitere Eigenschaften der local_mins

# local_min_data, prop_data = find_peaks(x = RF['Intensity'], prominence=0.2)
# local_min_smooth, prop_smooth = find_peaks(x = SmoothRF['Intensity'], height = 0.05, prominence= 0.1)

# Peaks müssen größer sein, als 2% der Daten um sie herum
substancialPeakDemand = int(len(SmoothRF) * 0.015)

local_max_smooth_aggr = argrelmin(SmoothRF['Intensity'].to_numpy(), order = substancialPeakDemand)
indexList = np.asarray(local_max_smooth_aggr)[0]

peaks = SmoothRF[SmoothRF.index.isin(indexList)]

# # Manuelles Hinzufügen der über- und untersteuerten Peaks
# # Indizes der Peaks mit Ordnungen -1, 0 und 1 genormt auf den degree of smoothnes der Daten
# indizes = [[int(3805/dgs), int(3980/dgs)], [int(4020/dgs), int(4460/dgs)], [int(4510/dgs), int(4670/dgs)]]

# for r in range(0,3,1):
#     # Filter die Werte raus, die über 97% Intensität haben (also likely Teil der Peaks sind)
#     values = SmoothRF['Intensity'][indizes[r][0]:indizes[r][1]].loc[lambda x : x>0.97]
#     # Positionen dieser Werte raussuchen
#     positions = SmoothRF['position'][values.index] 

#     # Mittelwert der Position bilden
#     meanpos = np.mean(positions)
#     std = np.std(positions, ddof=1)
#     deltapos = std * np.sqrt(len(positions))

#     meanint = np.mean(values)
#     std = np.std(values, ddof=1)
#     deltaint = std * np.sqrt(len(positions))
#     peaks = pd.concat([peaks, pd.DataFrame({'position': [meanpos], 'dPos': [deltapos], 'Intensity': [meanint], 'dInt': [deltaint]})])

# ERSTE ZWEI PEAKS ENTFERNEN, WEIL SIE SHUTTER SIND
peaks = peaks.iloc[2:]

peaks = peaks.sort_values('position')


# Werte abspeichern
peaks.to_csv('O8/Einzelspalt.csv', sep=';', index = False)

plt.plot(peaks['position'],  peaks['Intensity'], 
        label = "Minima der geglätteten Daten", color = 'lightgreen', linestyle='None', marker='o', markersize=8)
# plt.plot(SmoothRF['position'][local_min_smooth],  SmoothRF['Intensity'][local_min_smooth], label = "Maximum der smoothed Data mit find_peaks", color = 'lightgreen', linestyle='None', marker='o', markersize=4)

# Smoothed Data
ax.errorbar(x = SmoothRF['position'], y = SmoothRF['Intensity'], 
         label = "geglättete Daten - je " + str(dgs) + " Pixel zusammengefasst", 
         color = 'red', linestyle='None', marker='o',  markersize=2)

#,xerr = SmoothRF['dPos'], yerr = SmoothRF['dInt'],  markersize=6, capsize=3, elinewidth = 0.5 


#Messwerte plotten
ax.errorbar(x_data, y_data, label = 'Intensität des Lichtes Einzelspalt', 
            color = 'mediumblue', linestyle='None', marker='o', capsize=3, markersize=1, elinewidth = 0.5 )


################

# cosmetics

plt.xlabel('Position $x$ in cm',fontsize=fnt)
plt.ylabel('Intensität in % des maximalen Grauwertes', fontsize=fnt)
plt.legend(fontsize=fnt, loc='upper left') #Legende printen
plt.title("Intensitätsverteilung Einzelspalt", fontsize=fnt)
plt.grid()
plt.xticks(fontsize=fnt)
plt.yticks(fontsize=fnt)

plt.savefig("O8/IntensitatEinzelspalt.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5) 
plt.show()
