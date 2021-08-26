import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# Model Parameters
pop1 = 10
pop2 = 10
strat1 = 2
strat2 = -2

time = 300
KM = 100
r1 = 0.25
r2 = 0.25
c = 0.25
k = [.5,.5]
IC = [pop1,pop2,strat1,strat2]

sk = 2
sa = 4
sb = 10
sr = 10 #Balance between natural and sexual selection

bM = 0.15 #.5
rM = 0.25

def evoLV(X, t):

    x = X[0]
    y = X[1]
    u = X[2]
    mu = X[3]

    #sb = 10*(1-.0034)**t
    #sr = 10*(1-.004)**t

    K = KM*math.exp(-(u**2)/sk)

    b1 = bM*math.exp(-((u-mu)**2)/sb)
    b2 = bM*math.exp(-((mu-u)**2)/sb)
    r1 = rM*math.exp(-(u-mu)**2/sr)

    dxdt = x * (r1/K*(K-x)-b1*y)
    dydt = y * (r2*(1-(y/(c*b2*x))))

    dKdv =  (-2*u*KM/sk)*math.exp(-(u**2)/sk)
    db1dv = (-2*bM*(u-mu)/sb)*math.exp(-((u-mu)**2)/sb)
    db2dv = (-2*bM*(mu-u)/sb)*math.exp(-((mu-u)**2)/sb)

    drdv = 0 #(-2*rM*(u-mu)/sr)*math.exp(-((u-mu)**2)/sr)

    dG1dv = r1/K*dKdv+(K-x)/(K**2)*(K*drdv-r1*dKdv)-y*db1dv
    dG2dv = r2*y*db2dv/(c*x*b2**2)

    dudt = k[0] * dG1dv
    dmudt = k[1] * dG2dv

    dxvdt = np.array([dxdt, dydt, dudt, dmudt])
    return dxvdt

intxv = np.array(IC)
pop = odeint(evoLV, intxv, range(time+1))

print ('Population Prey: %f' %pop[time][0])
print ('Strategy Prey: %f' %pop[time][2])

print ('Population Predator: %f' %pop[time][1])
print ('Strategy Predator: %f' %pop[time][3])

#Obtaining extinction times
for i in range(len(pop[:,1])):
    if (pop[:,1][i])<2:
        print(i)
        print('Predator')
        break

for j in range(len(pop[:,0])):
    if (pop[:,0][j])<2:
        print(j)
        print('Prey')
        break

plt.figure()
plt.subplot(211)
plt.title('Predator-Prey Dynamics: Control')
plt.plot(pop[:,0],label='Prey')
plt.plot(pop[:,1],label='Predator')
#plt.ylim(ymax=50)
plt.legend()
plt.grid(True)
plt.ylabel('Pop Density')
plt.subplot(212)
plt.plot(pop[:,2],label='k = ' + str(k[0]))
plt.plot(pop[:,3],label='k = ' + str(k[1]))
plt.grid(True)
plt.ylabel('Indv Strategy')
plt.show()

time_G = time

prey = []
pred = []

def predG(mu,time_G):
    x = pop[time_G][0]
    y = pop[time_G][1]
    u = pop[time_G][2]
    b2 = bM*math.exp(-(((mu-u)**2)/sb))
    Gfunc2 = r2*(1-(y/(c*b2*x)))
    if Gfunc2<-1:
        Gfunc2=-1
    return Gfunc2

def preyG(u,time_G):
    x = pop[time_G][0]
    y = pop[time_G][1]
    mu = pop[time_G][3]
    b1 = bM*math.exp(-(((u-mu)**2)/sb))
    K = KM*math.exp(-(u ** 2)/sk)
    Gfunc1 = r1/K*(K-x)-b1*y
    if Gfunc1<-1:
        Gfunc1=-1
    return Gfunc1

scale=4

xp = np.arange(-scale, scale, .1)
yp = np.arange(0, time+1, 1)
Xp, Yp = np.meshgrid(xp, yp)

for i in yp:
    temp1 = []
    temp2 = []
    for j in xp:
        temp1.append(predG(j,i))
        temp2.append(preyG(j,i))
    pred.append(temp1)
    prey.append(temp2)
    temp1 = temp2 = []

G_pred=[]
G_prey=[]
for i in yp:
    j = pop[i][3]
    G_pred.append(predG(j,i))

for i in yp:
    j = pop[i][2]
    G_prey.append(preyG(j,i))

prey = np.array(prey)
pred = np.array(pred)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Xp, Yp, prey, cmap='Blues')
ax.plot_surface(Xp, Yp, pred, cmap='Oranges')
ax.plot3D(pop[:,3],yp,G_pred,c='gold') #fast
ax.plot3D(pop[:,2],yp,G_prey,c='purple') #slow
ax.set_xlabel('Evolutionary Strategy: v')
ax.set_ylabel('Time')
ax.set_zlabel('Fitness: G')
#ax.set_zlim(-1,0.1)
ax.view_init(35, 45)
plt.title('3D Adaptive Landscape: Predator-Prey Control',pad='20')
plt.show()
