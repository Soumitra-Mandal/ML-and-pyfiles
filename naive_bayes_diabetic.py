# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:20:42 2019

@author: Soumitra
"""












"""NAVIER BAYE'S TECHNIQUE
based on conditional probability
P(A|B)=P(B|A).P(A)/P(B)

"""

#NAVIER BAYES churn_modelling
dat=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning\Churn_Modelling.csv")
dat=dat.drop(["RowNumber","CustomerId","Surname"],axis=1)
dat.dtypes

#encoding geography and gender
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
dat.Geography=le1.fit_transform(dat.Geography)

dat.Gender=le1.fit_transform(dat.Gender)

ip=dat.drop(["Exited"],axis=1)
op=dat['Exited']

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

#Navier bsayes diabetic data
import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv(r"C:\Users\Lenovo\Desktop\Datasets\diabetic_data.csv")
dcopy=df.copy()
df.isnull().sum()
df.columns
df.dtypes
df.describe(include="all")
for value in ['race', 'gender', 'age','payer_code','medical_specialty','diag_1',
       'diag_2', 'diag_3','max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']:
    df[value].replace(['?'],[df.describe(include='all')[value][2]],inplace=True)
    
df['level1_diag1']=df['diag_1']
df.loc[df['diag_1'].str.contains('V'),['level1_diag1']]=0
df.loc[df['diag_1'].str.contains('E'),['level1_diag1']]=0
df['level2_diag2']=df['diag_2']
df.loc[df['diag_2'].str.contains('V'),['level2_diag2']]=0
df.loc[df['diag_2'].str.contains('E'),['level2_diag2']]=0
df['level3_diag3']=df['diag_3']
df.loc[df['diag_3'].str.contains('V'),['level3_diag3']]=0
df.loc[df['diag_3'].str.contains('E'),['level3_diag3']]=0
df=df.drop(["payer_code","encounter_id","patient_nbr","weight","medical_specialty",'admission_type_id',
       'discharge_disposition_id', 'admission_source_id'],axis=1)
dcopy=dcopy.drop(["payer_code","encounter_id","patient_nbr","weight","medical_specialty",'admission_type_id',
       'discharge_disposition_id', 'admission_source_id'],axis=1)






df = df.rename(columns={'glyburide-metformin': 'glyburide_metformin', 'glipizide-metformin': 'glipizide_metformin','glimepiride-pioglitazone': 'glimepiride_pioglitazone','metformin-rosiglitazone': 'metformin_rosiglitazone','metformin-pioglitazone':'metformin_pioglitazone'})

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
df.age=le1.fit_transform(df.age)
df.race=le1.fit_transform(df.race)
df.gender=le1.fit_transform(df.gender)
df.max_glu_serum=le1.fit_transform(df.max_glu_serum)
df.A1Cresult=le1.fit_transform(df.A1Cresult)
df.metformin=le1.fit_transform(df.metformin)
df.repaglinide=le1.fit_transform(df.repaglinide)
df.nateglinide=le1.fit_transform(df.nateglinide)
df.chlorpropamide=le1.fit_transform(df.chlorpropamide)
df.glimepiride=le1.fit_transform(df.glimepiride)
df.acetohexamide=le1.fit_transform(df.acetohexamide)
df.glipizide=le1.fit_transform(df.glipizide)
df.glyburide=le1.fit_transform(df.glyburide)
df.tolbutamide=le1.fit_transform(df.tolbutamide)
df.pioglitazone=le1.fit_transform(df.pioglitazone)
df.rosiglitazone=le1.fit_transform(df.rosiglitazone)
df.acarbose=le1.fit_transform(df.acarbose)
df.miglitol=le1.fit_transform(df.miglitol)
df.troglitazone=le1.fit_transform(df.troglitazone)
df.tolazamide=le1.fit_transform(df.tolazamide)
df.examide=le1.fit_transform(df.examide)
df.citoglipton=le1.fit_transform(df.citoglipton)
df.insulin=le1.fit_transform(df.insulin)
df.glyburide_metformin=le1.fit_transform(df.glyburide_metformin)
df.glipizide_metformin=le1.fit_transform(df.glipizide_metformin)
df.glimepiride_pioglitazone=le1.fit_transform(df.glimepiride_pioglitazone)
df.metformin_rosiglitazone=le1.fit_transform(df.metformin_rosiglitazone)
df.metformin_pioglitazone=le1.fit_transform(df.metformin_pioglitazone)
df.change=le1.fit_transform(df.change)
df.diabetesMed=le1.fit_transform(df.diabetesMed)
df.readmitted=le1.fit_transform(df.readmitted)
ip=df.drop(['diabetesMed','diag_1','diag_2','diag_3'],axis=1)
op=df['diabetesMed']


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




"""
K NEAREST NEIGHBOR ALGORITHM


"""
import pandas as pd
ds=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning\iris.csv.data",header=None)
ds.columns=["sl","sw","pl","pw","Class"]

from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
ds.Class=le1.fit_transform(ds.Class)

ip=ds.drop(["Class"],axis=1)
op=ds["Class"]

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(xtr,ytr)
kn=classifier.predict(xts)
print(kn)
accuracy=classifier.score(xts,yts)
print(accuracy)

from sklearn import metrics
recall=metrics.recall_score(yts,kn,average='micro')
print(recall)



import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

dn=datasets.load_iris()
X=dn.data
Y=dn.target

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(X,Y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

neighbors=np.arange(1,9)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtr,ytr)
    train_accuracy[i]=knn.score(xtr,ytr)
    test_accuracy[i]=knn.score(xts,yts)

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors,test_accuracy,label="Testing Acuuracy")
plt.plot(neighbors,train_accuracy,label="Training Acuuracy")
plt.legend()
plt.xlabel('Number of Neigbors')
plt.ylabel('Accuracy')
plt.show()

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(xtr,ytr)
knn.score(xts,yts)

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(xts)
confusion_matrix(yts,y_pred)


y_pred_proba=knn.predict_proba(xts)[:,1]
print(y_pred_proba)