# -*- coding: utf-8 -*-
"""
Created on Tue May 26 02:44:18 2020

@author: Rajat Gupta
website:
https://nbviewer.jupyter.org/gist/wpla/435437ddc5bcb1f6bdcae274117725e7
"""
#%%
%pylab inline
import pandas as pd
import scipy.stats
#%%
y = pd.read_excel("C:/Users/Rajat Gupta/Desktop/Interest_Rate_Model_Proj2/yield.xlsx",  "Tabelle1", index_col=0)
y
#%%
Z = []
logZ = []
for t in y.index:
    Z.append(exp(-y["yield"][t]))
    logZ.append(-y["yield"][t])
Z = pd.Series(Z, index=y.index)
logZ = pd.Series(logZ, index=y.index)
#%%
# Model parameters

a = 10   # Speed of reversion
c = 0.0002  # Volatility
r = 0.0075   # spot interest rates

gamma = 1/a
#%%
fig, ax = plt.subplots()
ax.plot(y.index, y["yield"], 'o-')
ax.set(xlabel='time (y)', ylabel='yield (%)',
       title='Yield curve')
ax.grid()
plt.show()
#%%
def make_spline(x, a):
    n = len(x) - 1
    h = []
    alpha = zeros(n)
    for i in range(n):
        h.append(x[i+1] - x[i])

    for i in range(1, n):
        alpha[i] = 3 / h[i] * (a[i+1] - a[i]) - 3 / h[i-1] * (a[i] - a[i-1])

    l = zeros(n+1)
    l[0] = 1
    mu = zeros(n+1)
    z = zeros(n+1)
    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1
    z[n] = 0
    b = zeros(n)
    c = zeros(n+1)
    d = zeros(n)

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j])/(3*h[j])
        
    def spline(t):
        if t < x[0]:
            t = x[0]
        if t >= x[-1]:
            t = x[-1]
        for j in range(n):
            if t >= x[j] and t <= x[j+1]:
                s = a[j] + b[j]*(t-x[j]) + c[j]*(t - x[j])**2 + d[j]*(t - x[j])**3
                s1 = b[j] + 2 * c[j] * (t - x[j]) + 3 * d[j] * (t - x[j])**2
                s2 = 2 * c[j] + 6 * d[j] * (t - x[j])
                return (s, s1, s2)
        return None
    return spline
#%%
yield_spline = make_spline(array(y.index), array(y["yield"]))

T = arange(y.index[0], y.index[-1], 1)
S = []
for t in T:
    (s, s1, s2) = yield_spline(t)
    S.append(s)
    
fig, ax = plt.subplots()
ax.plot(y.index, y["yield"], 'o', T, S)
ax.set(xlabel='time (y)', ylabel='yield (%)',
       title='Yield curve (Spline interpolated)')
ax.grid()
plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(y.index, Z, 'o-')

ax.set(xlabel='time (y)', ylabel='Z',
       title='zero coupon prices')
ax.grid()
plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(y.index, logZ, 'o-')

ax.set(xlabel='time (y)', ylabel='log(Z)',
       title='log Z')
ax.grid()
plt.show()
#%%
Z_spline = make_spline(array(y.index), array(Z))
logZ_spline = make_spline(array(y.index), array(logZ))

T = arange(y.index[0], y.index[-1], 1)
S = []
S1 = []
S2 = []
for t in T:
    (s, s1, s2) = logZ_spline(t)
    S.append(s)
    S1.append(s1)
    S2.append(s2)
    
fig, ax = plt.subplots()
ax.plot(y.index, logZ, 'o', T, S)
ax.set(xlabel='time (y)', ylabel='log Z',
       title='log Z (Spline interpolated)')
ax.grid()
plt.show()
#%%
T = arange(y.index[0], y.index[-1], 1)
S1 = []
for t in T:
    (s, s1, s2) = logZ_spline(t)
    S1.append(-s1)
    
fig, ax = plt.subplots()
ax.plot(T, S1, y.index, y["yield"])
ax.set(xlabel='time (y)', ylabel='f',
       title='Inst. forward rates')
ax.grid()
plt.show()
#%%

def eta(t):
    (s, s1, s2) = logZ_spline(t)
    return -s2 - gamma*s1 + c**2/(2*gamma)*(1-np.exp(-2*gamma*t))

T = arange(y.index[0], y.index[-1], 1)
E = []
for t in T:
    E.append(eta(t))
    
fig, ax = plt.subplots()
ax.plot(T, E)
ax.set(xlabel='time (y)', ylabel='eta(t)',
       title='Eta')
ax.grid()
plt.show()
#%%
T = arange(y.index[0], y.index[-1], 1)
r_mean = []
for t in T:
    r_mean.append(eta(t)*a)
    
fig, ax = plt.subplots()
ax.plot(T, r_mean)
ax.set(xlabel='time (y)', ylabel='eta(t)',
       title='r_mean')
ax.grid()
plt.show()
#%%
dt = 1

times = arange(y.index[0], y.index[-1], dt)
rates = []

rates = []
r_ = r 
for t in times:
    dr = gamma * (eta(t) - r_) * dt + c * np.random.normal()
    r_ = r_ + dr
    rates.append(r_)

fig, ax = plt.subplots()
ax.plot(times, rates)
ax.set(xlabel='time (y)', ylabel='r',
       title='interest rates')
ax.grid()
plt.show()
print(rates)
#%%

import numpy as np
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 16,8

def HW(r0, K, sigma, T=1., N=10, q=64, seed=777):    
    np.random.seed(seed)
    dt = T/float(N)    
    rates = [r0]
    for t in range(q):
        dr = K*(eta(t)-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

if __name__ == "__main__":
    x, y = HW(0.007595, 0.1, 0.000251, 0.5, 64)

    import matplotlib.pyplot as plt
    plt.plot(x,y)
    plt.show()
    z= x,y
#%%
z
