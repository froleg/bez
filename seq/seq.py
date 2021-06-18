import numpy as np
import scipy as sp
from scipy import special
from matplotlib import pyplot as plt
import yaml
from h20 import load_params
def bz_coef (d,P,t):
    J=[]
    for i in range(d+1):
        J.append(special.binom(d,i)*(t**i)*(1-t)**(d-i))
    return np.sum(np.array(J,dtype=np.float64)*P)
def bz_1d_coef (d,P,t):
    J=[]
    for i in range(d):
        J.append(special.binom(d-1,i)*(t**i)*(1-t)**(d-1-i)*(P[i+1]-P[i]))
    return np.sum(np.array(J,dtype=np.float64))*d
def bz_2d_coef (d,P,t):
    J=[]
    for i in range(d-1):
        J.append(special.binom(d-2,i)*(t**i)*(1-t)**(d-2-i)*(P[i+2]-2*P[i+1]+P[i]))
    return np.sum(np.array(J,dtype=np.float64))*d*(d-1)
def curvt(d,P,T):
#   bp=np.array([[bz_coef(d,P[0],x) for x in T],[bz_coef(d,P[1],x) for x in T]])
   b1dp=np.array([[bz_1d_coef(d,P[0],x) for x in T],[bz_1d_coef(d,P[1],x) for x in T]])
   b2dp=np.array([[bz_2d_coef(d,P[0],x) for x in T],[bz_2d_coef(d,P[1],x) for x in T]])
   return np.sqrt(np.abs(b2dp[1]*b1dp[0] - b2dp[0]*b1dp[1])/np.sqrt(b1dp[0]**2+b1dp[1]**2))
lp=load_params()
P=np.array(lp.P,dtype=np.float64)
N=lp.N
d=lp.d
In=lp.st
End=lp.end
tpt=np.linspace(In,End,N)
dt=tpt[1]-tpt[0]
cur=curvt(d,P,tpt)
print(cur)
intf=np.array([dt*0.5*(cur[x]+cur[x-1]) for x in range(0,N) if x>0])
reverse=np.flip(intf)
intf=np.flip(np.append(reverse,0.0))
insm=np.array([np.sum(intf[:x]) for x in range(1,N+1)],dtype=np.float64)
print(intf)
print(insm)
er=lp.er
InMx=(insm[len(insm)-1])
m=np.int(np.ceil(InMx*1/np.sqrt(8*er))+1)
print(m)
IntDist=np.linspace(0.0,InMx,m)
print(IntDist[m-2])
print(np.searchsorted(insm,IntDist[m-2]))
tOp=[In]
for i in range(1,m-1):
    inUp=np.searchsorted(insm,IntDist[i])
    el=((IntDist[i]-insm[inUp-1])/(insm[inUp]-insm[inUp-1]))*dt+tpt[inUp-1]
    tOp.append(el)
tOp.append(End)
topt=np.array(tOp,dtype=np.float64)
print(topt)
print(insm[len(insm)-1])
optPt=np.array([[bz_coef(d,P[0],x) for x in topt],[bz_coef(d,P[1],x) for x in topt]],dtype=np.float64)
print(optPt)
plt.plot(optPt[0],optPt[1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
