import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# Model Parameters
pop1 = 10 #prey 1
pop2 = 10 #prey 2
pop3 = 10 #predator
strat1 = 2.1 #prey 1 strategy
strat2 = 2 #prey 2 strategy
strat3 = 0 #predator strategy

time = 1000
KM = 100
r1 = 0.25
r2 = 0.25
r3 = 0.25
c = 0.25
k = [.5,.5,.5]

IC = [pop1,pop2,pop3,strat1,strat2,strat3]

sk = 2
sa = 10
sb = 10
sr = 10

bM = 0.5
rM = 0.25

def evoLV(X, t):

    x1 = X[0]
    x2 = X[1]
    y = X[2]
    u1 = X[3]
    u2 = X[4]
    mu = X[5]

    K1 = KM*math.exp(-(u1**2)/sk) #set this to just KM (and r1 constant) to see Red-Queen like dynamics (w/changed K above)
    K2 = KM*math.exp(-(u2**2)/sk) #set this to just KM (and r1 constant) to see Red-Queen like dynamics (w/changed K above)

    b1 = bM*math.exp(-((u1-mu)**2)/sb)
    b2 = bM*math.exp(-((u2-mu)**2)/sb)

    r1 = rM#*math.exp(-(u1-mu)**2/sr)
    r2 = rM#*math.exp(-(u2-mu)**2/sr)

    a1 = math.exp(-(u1-u2)**2/sa)
    a2 = math.exp(-(u2-u1)**2/sa)

    dx1dt = x1 * (r1/K1*(K1-x1-a1*x2)-b1*y)
    dx2dt = x2 * (r2/K2*(K2-x2-a2*x1)-b2*y)
    dydt = y * (r3*(1-(y/(c*(b1*x1+b2*x2)))))

    dK1dv = (-2*u1*KM/sk)*math.exp(-(u1**2)/sk)
    dK2dv = (-2*u2*KM/sk)*math.exp(-(u2**2)/sk)

    db1dv = (-2*bM*(u1-mu)/sb)*math.exp(-((u1-mu)**2)/sb)
    db2dv = (-2*bM*(u2-mu)/sb)*math.exp(-((u2-mu)**2)/sb)

    db3dv = (-2*bM*(mu-u1)/sb)*math.exp(-((mu-u1)**2)/sb)
    db4dv = (-2*bM*(mu-u2)/sb)*math.exp(-((mu-u2)**2)/sb)

    da1dv = (-2*(u1-u2)/sa)*math.exp(-((u1-u2)**2)/sa)
    da2dv = (-2*(u2-u1)/sa)*math.exp(-((u2-u1)**2)/sa)

    dr1dv = 0#(-2*rM*(u1-mu)/sr)*math.exp(-((u1-mu)**2)/sr)
    dr2dv = 0#(-2*rM*(u2-mu)/sr)*math.exp(-((u2-mu)**2)/sr)

    dG1dv = r1/K1*(dK1dv-da1dv*x2)+(K1-x1-a1*x2)/(K1**2)*(K1*dr1dv-r1*dK1dv)-y*db1dv
    dG2dv = r2/K2*(dK2dv-da2dv*x1)+(K2-x2-a2*x1)/(K2**2)*(K2*dr2dv-r2*dK2dv)-y*db2dv
    dG3dv = (r3*y/c)*(x1*db3dv+x2*db4dv)/((b1*x1+b2*x2)**2)

    du1dt = k[0] * dG1dv
    du2dt = k[1] * dG2dv
    dmudt = k[2] * dG3dv

    dxvdt = np.array([dx1dt,dx2dt,dydt, du1dt,du2dt, dmudt])
    return dxvdt

intxv = np.array(IC)
pop = odeint(evoLV, intxv, range(time+1))

print ('Population Prey1: %f' %pop[time][0])
print ('Strategy Prey1: %f' %pop[time][3])

print ('Population Prey2: %f' %pop[time][1])
print ('Strategy Prey2: %f' %pop[time][4])

print ('Population Predator: %f' %pop[time][2])
print ('Strategy Predator: %f' %pop[time][5])

plt.figure()
plt.subplot(211)
plt.title('Predator-Prey Dynamics: Speciation')
plt.plot(pop[:,0],label='Prey 1')
plt.plot(pop[:,1],label='Prey 2')
plt.plot(pop[:,2],label='Predator')
plt.legend()
#plt.ylim(ymax=50)
plt.grid(True)
plt.ylabel('Pop Density')
plt.subplot(212)
plt.plot(pop[:,3],label='k = ' + str(k[0]))
plt.plot(pop[:,4],label='k = ' + str(k[1]))
plt.plot(pop[:,5],label='k = ' + str(k[2]))
plt.grid(True)
plt.ylabel('Indv Strategy')
plt.show()

time_G = 1000

prey1 = []
prey2 = []
pred = []

def prey1G(u1,time_G):
    x1 = pop[time_G][0]
    x2 = pop[time_G][1]
    y = pop[time_G][2]
    u2 = pop[time_G][4]
    mu = pop[time_G][5]
    r1 = rM#*math.exp(-(u1-mu)**2/sr)
    a1 = math.exp(-(u1-u2)**2/sa)
    b1 = bM*math.exp(-((u1-mu)**2)/sb)
    K1 = KM*math.exp(-(u1**2)/sk) #set this to just KM (and r1 constant) to see Red-Queen like dynamics (w/changed K above)
    prey1 = r1/K1*(K1-x1-a1*x2)-b1*y
    if prey1<-.06:
        prey1=-.06
    if prey1>0.05:
        prey1=0.05
    return prey1

def prey2G(u2,time_G):
    x1 = pop[time_G][0]
    x2 = pop[time_G][1]
    y = pop[time_G][2]
    u1 = pop[time_G][3]
    mu = pop[time_G][5]
    r2 = rM#*math.exp(-(u2-mu)**2/sr)
    a2 = math.exp(-(u2-u1)**2/sa)
    b2 = bM*math.exp(-((u2-mu)**2)/sb)
    K2 = KM * math.exp(-(u2 ** 2) / sk)
    prey2 = r2/K2*(K2-x2-a2*x1)-b2*y
    if prey2 > 0.05:
        prey2 = 0.05
    if prey2 <-.06:
        prey2 = -.06
    return prey2

def predG(mu,time_G):
    x1 = pop[time_G][0]
    x2 = pop[time_G][1]
    y = pop[time_G][2]
    u1 = pop[time_G][3]
    u2 = pop[time_G][4]
    b1 = bM*math.exp(-((u1-mu)**2)/sb)
    b2 = bM*math.exp(-((u2-mu)**2)/sb)
    pred = r3*(1-(y/(c*(b1*x1+b2*x2))))
    if pred < -.06:
        pred = -.06
    return pred

scale=2.5

xp = np.arange(-scale, scale, .1)
yp = np.arange(0, time+1, 1)
Xp, Yp = np.meshgrid(xp, yp)

for i in yp:
    temp1 = []
    temp2 = []
    temp3 = []
    for j in xp:
        temp1.append(predG(j,i))
        temp2.append(prey1G(j,i))
        temp3.append(prey2G(j,i))
    pred.append(temp1)
    prey1.append(temp2)
    prey2.append(temp3)
    temp1 = temp2 = temp3 = []

G_pred=[]
G_prey1=[]
G_prey2=[]
for i in yp:
    j = pop[i][5]
    G_pred.append(predG(j,i))

for i in yp:
    j = pop[i][3]
    G_prey1.append(prey1G(j,i))

for i in yp:
    j = pop[i][4]
    G_prey2.append(prey2G(j,i))

prey1 = np.array(prey1)
prey2 = np.array(prey2)
pred = np.array(pred)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Xp, Yp, prey1, cmap='Blues')
ax.plot_surface(Xp, Yp, prey2, cmap='Oranges')
ax.plot_surface(Xp, Yp, pred, cmap='Greens')
ax.plot3D(pop[:,5],yp,G_pred,c='darkgreen') #fast
ax.plot3D(pop[:,3],yp,G_prey1,c='purple') #slow
ax.plot3D(pop[:,4],yp,G_prey2,c='gold') #slow
ax.set_xlabel('Evolutionary Strategy: v')
ax.set_ylabel('Time')
ax.set_zlabel('Fitness: G')
ax.set_zlim(-.06,0.05)
ax.view_init(35, 45)
plt.title('3D Adaptive Landscape: Predator-Prey Speciation',pad='20')
plt.show()
