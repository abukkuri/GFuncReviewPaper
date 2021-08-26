import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

#For 2 species

# Model Parameters
pop= 20
strat1 = 0 #normal cell
strat2 = 0 #cancer cell

time = 1000
K = 100
r = 0.25
var = 0.5
cov = 0#.25

lam = 1
b = 0.3


IC = [pop,strat1,strat2]

def evoLV(X, t):

    if t>600: #800 and 300 and 500
        m1=0
        m2=0.3
    elif t>200:
        m1 = 0.3
        m2 = 0
    else:
        m1=m2=0

    x = X[0]
    u1 = X[1]
    u2 = X[2]

    if x<1:
        x=0

    dxdt = x * (r * (1-x/K)-m1/(lam+b*u1)-m2/(lam+b*u2))

    dG1dv  = b*m1/(b*u1 + lam)**2
    dG2dv = b*m2/(b*u2 + lam)**2

    dv1dt = var * dG1dv + cov * dG2dv
    dv2dt = var * dG2dv + cov * dG1dv

    dxvdt = np.array([dxdt, dv1dt, dv2dt])
    return dxvdt

intxv = np.array(IC)
pop = odeint(evoLV, intxv, range(time+1))

if cov>0:
    txt = 'Cross-Resistant Drugs'
elif cov == 0:
    txt = 'Independent Drugs'
elif cov<0:
    txt = 'Double Bind Drugs'

plt.figure()
plt.subplot(211)
plt.title(txt)
plt.plot(pop[:,0], label = 'Cancer Cell')
plt.legend()
plt.grid(True)
plt.ylabel('Pop Size, x')
plt.subplot(212)
plt.plot(pop[:,1],label='Drug 1')
plt.plot(pop[:,2],label='Drug 2')
plt.legend()
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Indv Strategy, v')
plt.show()