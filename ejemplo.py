import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def f(x,a):
    
    return np.sin(x*a)
# def f(x,a):
    
#     return x*a


real_a = 2

ndata = 4

sig = 0.5

t = np.arange(0,ndata)

y = np.array([f(ti,real_a) + sig*np.random.normal() for ti in t])

# %%%% plot function

tplot = np.arange(0,ndata,0.01)

yplot = [f(ti,real_a) for ti in tplot]

plt.figure(0)
plt.plot(tplot,yplot)
plt.scatter(t,y)


# %% likelihood

def like(a):
    
    aux = np.array([f(xi,a) for xi in t])
    
    aux2 = np.array(y)
    
    exponent = (((aux2 - aux)**2).sum())/(2*sig**2)
    
    down = (2*np.pi*sig)**(len(y)/2)
    
    ans = np.exp(-exponent)/down
    
    return ans
    

# %% plot likelihood

a_array = np.arange(-10,100,0.01)
lplot = np.array([like(a) for a in a_array])
plt.figure(1)
plt.plot(a_array,lplot)
    
# %%% posterior

def prior(a):
    
    prior_ = norm(loc=0,scale=2)
    
    return prior_.pdf(a)
    
    
def post(a):
    
    return like(a) * prior(a)


# %% plot post
    
    
    
a_array = np.arange(-10,15,0.01)
pplot = np.array([post(a) for a in a_array])
plt.figure(2)
plt.plot(a_array,pplot)
plt.show()






