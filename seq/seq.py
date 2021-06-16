import numpy as np
import scipy as sp
from scipy import special
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
   return np.sqrt(np.abs(b2dp[1]*b1dp[0] + b2dp[0]*b1dp[1])/np.sqrt(b1dp[0]**2+b1dp[1]**2))
P=np.array([[0,0.55,11.8,16.5,22,24.5,18],[0.0,6.9,9.2,6.25,3.2,0,-1.0]],dtype=np.float64)
N=102
d=6
In=0.0
End=1.0
tpt=np.linspace(In,End,N)
print(bz_coef(6,P[0],0.4))
print(bz_1d_coef(6,P[0],0.4))
print(bz_2d_coef(6,P[0],0.4))
print(curvt(d,P,tpt))
intf=np.array([np.sum(0.5*curvt(d,P,tpt)[:x]) for x in range(0,N)])
print(intf)