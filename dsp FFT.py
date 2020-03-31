# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:22:45 2019

@author: Soumitra
"""

import math
import numpy as np
import numpy.fft as f
import matplotlib.pyplot as plt
n = np.arange(12);
x = ((-1)**n)*(n+1)
plt.xlabel('n');
plt.ylabel('x[n]');
plt.title(r'Plot of DT signal x[n]');

plt.stem(n, x);


#dft
n = np.arange(12);
x = ((-1)**n)*(n+1)
y = f.fft(x)
print(y)

#magnitude vs frequency
import cmath as cm
p=[]
for i in range(12):
    p.append(cm.phase(y[i]))
m=[]
for i in range(12):
    m.append(abs(y[i]))
k = [0]
for i in range(11):
    k.append(((i+1)*math.pi)/12)
    
plt.xlabel('k');
plt.ylabel('magnitude');
plt.title(r'Plot of mag vs frequency');

plt.stem(k, m);
  
