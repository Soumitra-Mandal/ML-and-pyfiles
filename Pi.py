# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:35:21 2019

@author: Soumitra
"""

"""
3 + 4/(2*3*4)-4/(4*5*6)+....

"""
from math import pi as p

Pi=3
a=2
b=3
c=4
i=1
for j in range(1000000):
  div=a*b*c  
  Pi=Pi + i*(4/div) 
  a=a+2
  b=b+2
  c=c+2
  acc=p-Pi
  i*=-1

print(Pi)

