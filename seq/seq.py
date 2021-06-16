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
P=np.array([0,0.55,11.8,16.5,22,24.5,18],dtype=np.float64)
print(bz_coef(6,P,0.4))
print(bz_1d_coef(6,P,0.4))
print(bz_2d_coef(6,P,0.4))