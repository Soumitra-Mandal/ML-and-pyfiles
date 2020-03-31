# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:13:33 2020

@author: Soumitra
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dat=pd.read_excel(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\AirQualityUCI\AirQualityUCI.xlsx")
dat.isnull().any()
for value in dat.columns:
    print(value,":",sum(dat[value]==-200))
print(dat.columns)
for value in ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
       'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
       'PT08.S5(O3)', 'T', 'RH', 'AH']:
    dat[value].replace([[-200]],[0],inplace=True)

for value in ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
       'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
       'PT08.S5(O3)', 'T', 'RH', 'AH']:
    dat[value].replace([[0]],[dat.describe(include='all')[value][6]],inplace=True)

dat.drop(['Date','Time'],axis=1,inplace=True)
    
cor=dat.corr()
plt.figure(figsize=(12,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()

ip=dat.drop(["CO(GT)"],axis=1)
op=dat["CO(GT)"]

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts= train_test_split(ip,op,test_size=0.2)
from sklearn.linear_model import LinearRegression
alg=LinearRegression()

alg.fit(xtr,ytr)


print("m is ",alg.coef_)
print("c is ",alg.intercept_)

yp=alg.predict(xts)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
b=mean_squared_error(yts,yp)
a=r2_score(yts,yp)
accuracy=alg.score(xts,yts)
print(a)
print(b)
print(accuracy)




