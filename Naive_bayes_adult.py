# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:06:33 2019

@author: Lenovo
"""

"""
https://colab.research.google.com/notebooks/welcome.ipynb
"""
# adult data navier bayes
import pandas as pd
from sklearn import svm
dcopy=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\adult.data",header=None)
data=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\adult.data",header=None)
data.dtypes
for value in [1,3,5,6,7,8,9,13,14]:
     print(value,":",sum(data[value]==" ?"))
for value in [1,6,13]:
    data[value].replace([' ?'],[data.describe(include='all')[value][2]],inplace=True)    

data.columns=['a','b','c','d','e','f','g','h','I','J','k','i','j','K','l']
data.dtypes

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
data.b=le1.fit_transform(data.b)
data.d=le1.fit_transform(data.d)
data.f=le1.fit_transform(data.f) 
data.g=le1.fit_transform(data.g)
data.h=le1.fit_transform(data.h)
data.I=le1.fit_transform(data.I)
data.J=le1.fit_transform(data.J)
data.K=le1.fit_transform(data.K)
data.l=le1.fit_transform(data.l)


ip=data.drop(["l"],axis=1)
op=data['l']


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()
clf.fit(xtr,ytr)

#training
yp=clf.predict(xts)
accuracy=clf.score(xts,yts)
print(accuracy)


from sklearn import metrics
from sklearn.metrics import classification_report 
classification_report(yts,yp)
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)


#adult data SVM
import pandas as pd
dcopy=pd.read_csv(r"C:\Users\Lenovo\Desktop\asutosh\adult data.csv",header=None)
data=pd.read_csv(r"C:\Users\Lenovo\Desktop\asutosh\adult data.csv",header=None)
data.dtypes
for value in [1,3,5,6,7,8,9,13,14]:
     print(value,":",sum(data[value]==" ?"))
for value in [1,6,13]:
    data[value].replace([' ?'],[data.describe(include='all')[value][2]],inplace=True)    

data.columns=['a','b','c','d','e','f','g','h','I','J','k','i','j','K','l']
data.dtypes

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
data.b=le1.fit_transform(data.b)
data.d=le1.fit_transform(data.d)
data.f=le1.fit_transform(data.f)
data.g=le1.fit_transform(data.g)
data.h=le1.fit_transform(data.h)
data.I=le1.fit_transform(data.I)
data.J=le1.fit_transform(data.J)
data.K=le1.fit_transform(data.K)
data.l=le1.fit_transform(data.l)


ip=data.drop(["l"],axis=1)
op=data['l']


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

alg=svm.SVC(C=100,gamma=0.1)

alg.fit(xtr,ytr)
yp=alg.predict(xts)

accuracy=alg.score(xts,yts)
print(accuracy)


from sklearn import metrics
from sklearn.metrics import classification_report 
classification_report(yts,yp)
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)