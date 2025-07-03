import numpy as np  # Numpy array library
import pandas as pd # Pandas Dataframes um Messdaten zu speichern
import matplotlib.pyplot as plt # Plots erstellen
import matplotlib.mlab as mlab
import scipy.stats # Statistische Auswertung - Unsicherheiten werden automatisch eingerechnet
from scipy.optimize import curve_fit
from scipy.signal import argrelmax
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

local_max_smooth_aggr = argrelmax(SmoothRF['Intensity'].to_numpy(), order = substancialPeakDemand)
indexList = np.asarray(local_max_smooth_aggr)[0]

peaks = SmoothRF[SmoothRF.index.isin(indexList)]

peaks = peaks.sort_values('position')

# PEAKS ENTFERNEN, WEIL SIE CLUTTER SIND
peaks = peaks.iloc[4:]
peaks = peaks.iloc[:len(peaks)-4]

# Werte abspeichern
peaks.to_csv('O8/Lochblende.csv', sep=';', index = False)

#######################
# Positionen so verschieben, dass Maximum 0. Ordnung bei 0cm liegt

position = position - RF['position'].iloc[peaks.idxmax().iloc[2]*dgs]
peaks['position'] = peaks['position']- SmoothRF['position'].iloc[peaks.idxmax().iloc[2]]
SmoothRF['position'] = SmoothRF['position'] - SmoothRF['position'].iloc[peaks.idxmax().iloc[2]]

###################################################################################################
# Plot

fig, ax = plt.subplots()
# fig ist das eigentliche Bild, ax ist ein Datenobjeke


# Peaks plotten
ax.errorbar(peaks['position'],  peaks['Intensity'], 
        label = "Maxima der geglätteten Daten", 
        color = 'lightgreen', linestyle='None', marker='o', markersize=8)
 
# Smoothed Data plotten
ax.errorbar(x = SmoothRF['position'], y = SmoothRF['Intensity'], 
        label = "geglättete Daten - je " + str(dgs) + " Pixel zusammengefasst", 
        color = 'red', linestyle='None', marker='o', markersize=2, capsize=3, elinewidth = 0.5)
# xerr = SmoothRF['dPos'], yerr = SmoothRF['dInt'],

#Messwerte plotten
# ax.errorbar(np.array([value.n for value in position]),  Intensity, label = 'Intensität des Lichtes Lochblende', 
#             color = 'mediumblue', linestyle='None', marker='o', capsize=3, markersize=1, elinewidth = 0.5)



##########################
# PloFit für Hälfte der Maxima durchführen
# start = (peaks.idxmax()[2]+1)

x_data = SmoothRF['position']#[start:]
x_err = SmoothRF['dPos']#[start:]
y_data = SmoothRF['Intensity']#[start:]
y_err = SmoothRF['dInt']#[start:]


# # Fitfunktion definieren
#         # param B = Lochblenden durchmessen
#         # Kleiwinkelnäherung sin(alp)=tan(alp)= pos/SS = x/SS
# def fit_function(x, I, B):
#     return  I * (j1(1, np.pi*B*x / (lamb*SS.n)) / (np.pi*B*x / (lamb*2.0*SS.n)))**2


# # Curve-Fit mit Unsicherheiten in y
# params, covariance = curve_fit(fit_function, x_data, y_data, sigma = y_err) # 
# fit_errors = np.sqrt(np.diag(covariance))  # Fehler der Fit-Parameter
# I_value = params[0]
# I_error = fit_errors[0]
# B_value = params[1]
# B_error = fit_errors[1]

# dof = len(RF.index)-len(params)
# # chi2 = sum([((fit_function(x,I_value, B_value)-y)**2)/(u**2) for x,y,u in zip(x_data,y_data,y_err)])

# # Fit-Ergebnisse ausgeben
# # print(f"A = {A_value:.6f} ± {A_error:.6f}")
# # print(f"x0 = {x0_value:.6f} ± {x0_error:.6f}")
# # print(f"Chi-Quadrat/dof: {chi2/dof}")

# x_ax = np.linspace(-5, 5, 1000) 
# y_ax = fit_function(x_ax, I_value, B_value)


# # D ist Abstand SS zwischen Schrim und Lochblende
# #label = r'Fit: $I = \left( \frac{J_1(\theta \\ 2)}{\theta \\ 4}\right)^2$ mit $\theta = \frac{2 \pi \cdot B \cdot x}{\lambda \cdot D}$'  
# label = r'Fit: $I = \left( \frac{J_1(A \cdot x)}{ \frac{A \cdot x}{2} }\right)^2$ mit $\theta = \frac{2 \pi \cdot B \cdot x}{\lambda \cdot D}$'  
# label += f"\n $(I = {I_value:.6f} \\pm {I_error:.6f})$ cm"
# label += f"\n $(A = {B_value:.6f} \\pm {B_error:.6f})$ cm"


# Plot zeichnen
#plt.plot(x_ax, y_ax, label = label, linewidth = 2, color = 'lightblue')



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
