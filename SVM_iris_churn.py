# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:45:21 2019

@author: Lenovo
"""

"""
SUPPORT VECTOR MACHINE
discriminative classifier formally defined by a separating hyper plane.

kernel function:increase dimensionality of the data sets.
linear kernel fn
polynomial kfn
rbf(radius Bases kernel fn)
NPTEL lectures
"""
import pandas as pd
import numpy as np
from sklearn import svm,datasets
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data[:,:2]#we only take first two features
Y=iris.target



x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
h=(x_max/x_min)/100
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
X_plot=np.c_[xx.ravel(),yy.ravel()]

#Create SVC Model object
#SVM regulation parameter 
svc=svm.SVC(kernel='linear',C=100,gamma=0.01).fit(X,Y)
Z=svc.predict(X_plot)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx,yy,Z,cmap=plt.cm.tab10,alpha=0.6)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Set1)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(),xx.max())
plt.title("SVC with linear kernel")



svc=svm.SVC(kernel='rbf',C=100,gamma=0.01).fit(X,Y)
Z=svc.predict(X_plot)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx,yy,Z,cmap=plt.cm.tab10,alpha=0.6)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Set1)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(),xx.max())
plt.title("SVC with Rbf kernel")



svc=svm.SVC(kernel='poly',C=100,gamma=0.01).fit(X,Y)
Z=svc.predict(X_plot)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx,yy,Z,cmap=plt.cm.tab10,alpha=0.6)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Set1)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(),xx.max())
plt.title("SVC with Rbf kernel")


#churn modelling SVM
dt=pd.read_csv(r"C:\Users\Lenovo\Desktop\ai and ml dec\data\Churn_Modelling.csv")
dt.columns
dt.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1,inplace=True)

from sklearn.utils import shuffle
n_samples=2000
remove_list=[]
for j in range(dt.shape[0]):
    for i in ['Exited']:
        if dt[i][j]==0:
            remove_list.append(j)
remove_list=shuffle(remove_list)
remove_list=remove_list[n_samples:]

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
dt.Geography=le1.fit_transform(dt.Geography)




dt.Gender=le1.fit_transform(dt.Gender)
ip=dt.drop(["Exited"],axis=1)
op=dt['Exited']

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

from sklearn import svm
alg=svm.SVC(C=100,gamma=0.1)

alg.fit(xtr,ytr)
yp=alg.predict(xts)

accuracy=alg.score(xts,yts)
print(accuracy)
from sklearn import metrics
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)

