# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#1d array
import numpy as np
a=np.array([1,2,3])


#2d array
a=np.array([(1,2,3),(4,5,6)])
b=np.ones((3,4))
c=np.ones_like(a)
d=np.eye(5)
e=np.linspace(0,100,9)
f=np.arange(0,10,2.5)
f=np.full((3,3),6)
f=np.random.rand(4,5)*100
f=np.random.randint(10,size=(2,3))
f=np.random.normal(2,0.2,(50,2))
arr1=np.array(([1,2,3],[4,5,6]))
arr2=np.array([3,4,5])
f=np.add(arr1,arr2)
f=np.log(arr1)
f=np.floor(4.2)
f=np.mean(arr1)
f=arr1.mean()
g=arr1.shape
g=arr1.size
f=arr1.T
f=arr1.reshape(3,2)
g=np.insert(arr1,2,5)
g=np.delete(arr1,0,axis=1)
g[0,0]=2
g=arr1[:1,:1]
g=np.concatenate((a,arr1),axis=0)
#pandas
import pandas as pd
import numpy as np
xyz_web={'day': [1,2,3,4,5,6], 'visitors':[1000,700,650,400,1200,1100], 'bounce_rate':[40,50,100,55,60,80]}
df=pd.DataFrame(xyz_web)
print(df)
print(df.head(2))
df1=pd.DataFrame({'HPI':[80,90,100],'INT_RATE':[12,15,34],'index':[12,32,12]})
df2=pd.DataFrame({'HPI':[80,90,100],'INT_RATE':[22,18,64],'index':[15,72,10]})
merged=pd.merge(df1,df2,on='HPI')
print(merged)
df3=pd.DataFrame({'A':[23,43,56],'B':[54,77,90]}, index=[23,11,2])
df4=pd.DataFrame({'D':[82,49,96],'E':[50,75,97]}, index=[32,11,12])
joined=df4.join(df3)
print(joined)
df5=df.rename(columns={'visitors':'users'})
print(df5)
df.set_index("day",inplace=True)
print(df)
print(df.sum())
print(df.mean())
print(df.std())
print(df.count())
print(df.max())
print(df.min())
print(df.median())
print(df.cov())
df.iloc[0,:]
df.describe(include="all")
dfn=pd.DataFrame(np.random.randn(10,2),index=[0,9,5,3,7,4,13,12,1,6,],columns=['col1','col2'])
sorted_df=dfn.sort_index(ascending=False)
print(sorted_df)
print(sorted_df.sort_values(by="col1", ascending=False))
pd.concat([df,df1],axis=1)
df.append(df1)
data=pd.read_csv(r"C:\Users\Lenovo\Desktop\ai and ml dec\data\Churn_Modelling.csv")
data1=pd.read_csv(r"C:\Users\Lenovo\Desktop\iris.csv",header=None)
data1.columns=["SL","SW","PL","PW","CLASS"]
data2=pd.read_csv(r"C:\Users\Lenovo\Desktop\ai and ml dec\data\movie_metadata.csv")
data2.shape
data2.columns.str.upper()
data2.isnull()
data2.isnull().any()
data2.isnull().sum()
data2.isnull().sum().sum()
data3=data2.dropna(x=0)
data3.shape
data4=data2.fillna('x')
data4=data2.fillna(data2.mean())
data2.info()
df=pd.DataFrame(np.random.randn(6,3),index=["a","c","e","f","g","h"],columns=["one","two","three"])
df=df.reindex(['a','b','c','d','e','f','g','h'])
print(df["one"].notnull())
print(df["one"].isnull())
data5=pd.read_csv(r"C:\Users\Lenovo\Downloads\adult.data")