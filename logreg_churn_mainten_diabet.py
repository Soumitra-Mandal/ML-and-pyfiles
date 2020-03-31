# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:51:06 2019

@author: Soumitra
"""

#logistic regression
import pandas as pd
import numpy as np
d1=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\Churn_Modelling.csv")
d1=d1.drop(["RowNumber","CustomerId","Surname"],axis=1)
d1.dtypes
d1.isnull().any()
#encoding geography and gender
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
d1.Geography=le1.fit_transform(d1.Geography)

d1.Gender=le1.fit_transform(d1.Gender)

ip=d1.drop(["Exited"],axis=1)
op=d1['Exited']

from sklearn.preprocessing import OneHotEncoder

"""ohe=OneHotEncoder(categorical_features=[1])
ip=ohe.fit_transform(ip).toarray()    obsolete"""

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
ip = np.array(ct.fit_transform(ip), dtype=np.float)



from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

from sklearn.linear_model import LogisticRegression
alg=LogisticRegression()

#train the algorithm with training data
alg.fit(xtr,ytr)

accuracy=alg.score(xts,yts)
print(accuracy)

yp=alg.predict(xts)

from sklearn import metrics
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)

import numpy as np
ip=np.array([1,0,0,619,0,42,2,0,1,1,1,101349]).reshape(1,-1)
ip=sc.transform(ip)
alg.predict(ip)




#maintenance data logistic regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
dt=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\maintenance_data.csv")
dt.isnull().sum().sum()
dt.columns
dt.dtypes

dt.broken.mean()

#distplot
plt.figure(figsize=(12,5))
sns.distplot(dt.lifetime[dt.broken==1])
sns.distplot(dt.lifetime[dt.broken==0])
plt.legend(['broken'],['not broken'])
plt.show()

#swarmplot1
plt.figure(figsize=(12,5))
sns.swarmplot(x=dt.team,y=dt.lifetime,hue=dt.broken)
plt.ylim([50,90])
plt.show()

#swarmplot2
plt.figure(figsize=(12,5))
sns.swarmplot(x=dt.provider,y=dt.lifetime,hue=dt.broken)
plt.show()



dt.describe(include='all')
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()

dt.team=le1.fit_transform(dt.team)

dt.provider=le1.fit_transform(dt.provider)

ip=dt.drop(['broken'],axis=1)
op=dt['broken']
from sklearn.preprocessing import OneHotEncoder
"""ohe=OneHotEncoder(categorical_features=[4,5])
ip=ohe.fit_transform(ip).toarray()"""

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [4,5])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
ip = np.array(ct.fit_transform(ip), dtype=np.float)

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.4)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

from sklearn.linear_model import LogisticRegression
alg=LogisticRegression()

#train the algorithm with training data
alg.fit(xtr,ytr)

accuracy=alg.score(xts,yts)
print(accuracy)

yp=alg.predict(xts)

from sklearn import metrics
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)
import numpy as np
ip=np.array([1,0,0,0,0,0,1,56,92.1789,104.23,96.5172]).reshape(1,-1)
ip=sc.transform(ip)
alg.predict(ip)




#diabetes_data logistic regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\diabetic_data.csv")
dcopy=df.copy()
df.isnull().sum()
df.columns
df.dtypes
df.describe(include="all")

sns.countplot(df.diabetesMed)
plt.show()



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
"""from sklearn.utils import shuffle
n_samples=2000
remove_list=[]
for j in range(df.shape[0]):
    for i in ['diabetesMed']:
        if df[i][j]==0:
            remove_list.append(j)
remove_list=shuffle(remove_list)
remove_list=remove_list[n_samples:]
sns.countplot(df.diabetesMed)
plt.show()
"""
ip=df.drop(['diabetesMed','diag_1','diag_2','diag_3'],axis=1)
op=df['diabetesMed']

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0,1,2,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])

ip=ohe.fit_transform(ip).toarray()


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

from sklearn.linear_model import LogisticRegression
alg=LogisticRegression()
#train the algorithm with training data
alg.fit(xtr,ytr)

accuracy=alg.score(xts,yts)
print(accuracy)

yp=alg.predict(xts)

from sklearn import metrics
cm=metrics.confusion_matrix(yts,yp)
print(cm)

recall=metrics.recall_score(yts,yp)
print(recall)