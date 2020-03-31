# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:59:30 2019

@author: Soumitra
"""

"""
UNSUPERVISED LEARNING
:
1.CLUSTERING
    1.1. HIERCHICAL 
    1.2. K-MEANS
2.ANOMALY DETECTION
"""
#k MEANS clustering
import pandas as pd
data=pd.read_csv(r"C:\Users\Lenovo\Desktop\iris.csv")
data.columns=["SL","SW","PL","PW","CLS"]
ip=data.drop(['CLS'],axis=1)
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
data.CLS=le1.fit_transform(data.CLS)
from sklearn import cluster
km=cluster.KMeans(n_clusters=3)
km.fit(ip)
k=km.predict(ip)
print(k)
data["predict"]=k

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.scatter(data.SL,data.PL)
plt.show()

plt.figure(figsize=(12,5))
plt.scatter(data.SL,data.PL,c=km.labels_)
plt.show()

print(km)



import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X,Y_true=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
data2=pd.DataFrame(X,Y_true)
plt.scatter(X[:,0],X[:,1],s=50)
print(X)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
Y_means=kmeans.predict(X)
data2["predict"]=Y_means
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(X[:,0],X[:,1],c=Y_means,s=50,cmap="viridis")
plt.scatter(centroids[:,0],centroids[:,1],c="black",s=200,alpha=0.7)

def ClusterIndicesNumpy(ClustNum, labels_array):
    return np.where(labels_array == ClustNum)[0]

ClusterIndicesNumpy(2,kmeans.labels_)
X[ClusterIndicesNumpy(2,kmeans.labels_)]




"""
anomaly detection

"""

import matplotlib.pyplot as plt
x=np.random.randint(10,15,18)
y=np.random.randint(12,18,18)

data=np.zeros((18,2))
data[:,0]=x
data[:,1]=y

from sklearn import svm
alg=svm.OneClassSVM(nu=0.08,gamma=0.001)
alg.fit(data)
labels=alg.predict(data)
print(labels)

alg.predict(np.array([15,14]).reshape(1,2))

test=np.random.randint(15,35,(10,2))
lab2=alg.predict(test)
print(lab2)

plt.scatter(x,y,c=labels)
plt.scatter(test[:,0],test[:,1],c=lab2)
plt.xlim([2,42])
plt.ylim([2,42])
plt.show()