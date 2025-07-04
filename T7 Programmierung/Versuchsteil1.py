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

RF = pd.read_csv('RohdatenAufgabe1.csv', header=2)
plt.rcParams['figure.figsize'] = [19.2,10.8]

anzahlVersuchsreihen = 6
colors = [['mediumblue', 'cornflowerblue'],
          ['olivedrab', 'mediumaquamarine'],
          ['darkred', 'tomato'],
          ['chocolate', 'gold'],
          ['lightgreen','darkgreen'],
          ['pink','purple']]

fig, ax = plt.subplots()

def fit_function(x,A,x0):
    return A*x+x0

x_data=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
y_data=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
y_err=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
A_value=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
A_error=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
x0_value=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
x0_error=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
params=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
covariance=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
fit_errors=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
chi2=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
dof=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
chi2dof=[[0]*2,[0]*2,[0]*2,[0]*2,[0]*2,[0]*2]
h_1=[0,0,0,0,0,0]
h_2=[0,0,0,0,0,0]
kappa=[0,0,0,0,0,0]
k=[0,0,0,0,0,0]

for i in range(0,anzahlVersuchsreihen,1):
    ##### Berechnungen Unsicherheiten ####
    RF['dt_sys_MR'+str(i+1)]=0.0005*(RF['t_MR'+str(i+1)])+0.03 #sys. Unsicherheit aus Einführungspraktikum
    RF['dt_gross_MR'+str(i+1)]=0 #Größtfehlerabschätzung unserer Langsamkeit
    RF['dt_MR'+str(i+1)]=np.sqrt(RF['dt_sys_MR'+str(i+1)]**2+RF['dt_gross_MR'+str(i+1)]**2)
    RF['ut_MR'+str(i+1)]=unp.uarray(RF['t_MR'+str(i+1)],RF['dt_MR'+str(i+1)])
    
    RF['dh_sys_MR'+str(i+1)]=1 #systematische Unsicherheit Manometer
    RF['dh_gross_MR'+str(i+1)]=0 #Größtfehlerabschätzung Manometer
    RF['dh_MR'+str(i+1)]=np.sqrt(RF['dh_sys_MR'+str(i+1)]**2+RF['dh_gross_MR'+str(i+1)]**2)
    RF['uh_MR'+str(i+1)]=unp.uarray(RF['h_MR'+str(i+1)],RF['dh_MR'+str(i+1)])

    #### Datenpunkte ins Plot laden ####
    x_data_i = RF['t_MR'+str(i+1)]
    x_err_i = RF['dt_MR'+str(i+1)]
    y_data_i = RF['h_MR'+str(i+1)]
    y_err_i = RF['dh_MR'+str(i+1)]

    ax.errorbar(x_data_i, y_data_i, xerr=x_err_i, yerr=y_err_i, label='Versuchsreihe '+str(i+1), color=colors[i][1], linestyle='None', marker='o', capsize=8, markersize=9, elinewidth=2)
    
    #### Daten zum Berechnen zweier Geraden laden ####
    x_data[i][0] = RF['t_MR'+str(i+1)][1:5]
    y_data[i][0] = RF['h_MR'+str(i+1)][1:5]
    y_err[i][0] = RF['dh_MR'+str(i+1)][1:5]

    x_data[i][1] = RF['t_MR'+str(i+1)][7:11]
    y_data[i][1] = RF['h_MR'+str(i+1)][7:11]
    y_err[i][1] = RF['dh_MR'+str(i+1)][7:11]

    for j in range(0,2,1):
        params[i][j], covariance[i][j] = curve_fit(fit_function, x_data[i][j], y_data[i][j], sigma=y_err[i][j], absolute_sigma=True)
        fit_errors[i][j] = np.sqrt(np.diag(covariance[i][j]))
        A_value[i][j] = params[i][j][0]
        A_error[i][j] = fit_errors[i][j][0]
        x0_value[i][j] = params[i][j][1]
        x0_error[i][j] = fit_errors[i][j][1]
        dof[i][j] = len(RF.index)-len(params[i][j])
        chi2[i][j] = sum([((fit_function(x,A_value[i][j],x0_value[i][j])-y)**2)/(u**2) for x,y,u in zip(x_data[i][j],y_data[i][j],y_err[i][j])])
        chi2dof[i][j]=chi2[i][j]/dof[i][j]
        print("i =",i+1, " and j = ", j, " and A =", A_value[i][j], "+/-", A_error[i][j], "and x0 =", x0_value[i][j], "+/-", x0_error[i][j])
        print("i =",i+1, " and j = ", j, " and Chi-Quadrat/dof:", chi2dof[i][j])

    h_1[i] = RF['uh_MR'+str(i+1)][5]
    h_2[i] = fit_function(RF['t_MR'+str(i+1)][6],A_value[i][1],x0_value[i][1])
    #TO-DO: Unsicherheit h_2_i
    print("h_1_",str(i+1),"=", h_1[i])
    print("h_2_",str(i+1),"=", h_2[i])
    kappa[i]=h_1[i]/(h_1[i]-h_2[i])
    print("Kappa_",str(i+1),"=",kappa[i])
    
#### Mittelwert berechnen ####
mean = np.mean([value.nominal_value for value in kappa])
standab = np.std([value.nominal_value for value in kappa],ddof=1)
dmean = u.ufloat(mean,standab)
print("Mittelwert Kappa:", dmean)

#### Plot zeichnen ####
ax.set_xlim(-10,610)
ax.set_ylim(-0.5,30)
plt.xlabel("Zeit t in s",fontsize=20)
plt.ylabel("Höhe h in mm",fontsize=20)
plt.legend(fontsize=20)
plt.title("Höhe der Manometerflüssigkeit in mm in Abhängigkeit der Zeit",fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("Versuchsteil1.pdf", format='pdf', bbox_inches='tight', pad_inches=0.5)

plt.show()
