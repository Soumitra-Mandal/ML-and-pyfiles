# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:58:29 2019

@author: Soumitra
"""


"""
other ways to remove:
data2.dtypes data2.horsepowr.unique()
data2=data2[data2.horsepower!='?']
data2.horsepower=data2.horsepower.astype("float")

data2=data2.replace("?",np.NaN)
data2=data2.fillna("x")

"""


#auto mpg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
dp=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\auto-mpg.data-original",delim_whitespace=True,header=None)
dp.columns=["mpg", "cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
dp.isnull().sum()
dp.describe(include='all')
dp["mpg"].replace([np.nan],[dp.describe(include='all')["mpg"][4]],inplace=True)
dp["horsepower"].replace([np.nan],[dp.describe(include='all')["horsepower"][4]],inplace=True)

cor=dp.corr()
plt.figure(figsize=(12,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()

sns.pairplot(dp)
plt.show()

dp1=dp.drop(["car name","acceleration"],axis=1)
ip=dp1.drop(["mpg"],axis=1)
op=dp1["mpg"]


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts= train_test_split(ip,op,test_size=0.35)
from sklearn.linear_model import LinearRegression
alg=LinearRegression()

#training _algorithms
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