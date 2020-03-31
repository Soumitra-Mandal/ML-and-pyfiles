# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:49:10 2019

@author: Lenovo
"""
import pandas as pd
data=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\adult.data",header=None)
data.shape
data.isnull().any()
data.isnull().sum()
data.columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
for value in data.columns:
    print(value,":",sum(data[value]==" ?"))
data.describe(include="all")
for value in ['b','d','f','g','h','i','j','n','o']:
    data[value].replace([' ?'],[data.describe(include='all')[value][2]],inplace=True)
from matplotlib import pyplot as plt
plt.plot([1,2,3],[4,5,1])
plt.show()
X=[5,8,10]
Y=[2,6,6]
plt.plot(X,Y)
plt.title('info')
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.show()
#bar Graph
plt.bar([1,3,5,7,9],[5,2,7,8,2],label="Example One",color='orange')
plt.bar([2,4,6,8,10],[8,6,2,5,6],label="Example",color="g")
plt.legend()
plt.xlabel('Bar Number')
plt.ylabel('Bar Height')
plt.title("info")
plt.show()

#histogram
Height_in_cm=[22,55,35,45,65,75,85,63,26,51,41,15,23,32,55,44,46,72]
bins=[0,10,20,30,40,50,60,70,80,90]
plt.hist(Height_in_cm,bins,histtype="bar",rwidth=0.8,color="Blue")
plt.legend()
plt.xlabel("height")
plt.ylabel("Y")
plt.title("histogram")
plt.show()


#scatterplot
X=[1,2,3,4,5,6,7,8,9,10]
Y=[10,9,8,7,6,5,4,3,2,1]
plt.scatter(X,Y,label="skitscat",color="blue",s=10,marker="*")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.show()

#Area Plot
days=[1,2,3,4,5]
sleeping=[7,8,6,5,7]
eating=[2,3,2,1,3]
working=[7,8,9,8,10]
plt.plot([],[],color='m',label="sleeping",linewidth=5)
plt.plot([],[],color='c',label="eating",linewidth=5)
plt.plot([],[],color='r',label="working",linewidth=5)
plt.stackplot(days,sleeping,eating,working,colors=["m","c","r"])
plt.xlabel("day count")
plt.ylabel("activities")
plt.legend()
plt.show()

#piechart
slices = [7,2,13,1]
activities=["sleeping","eating","working","playing"]
cols=["c","m","r","b"]
plt.pie(slices,labels=activities,colors=cols,shadow=True,startangle=80,explode=(0,0.2,0,0),autopct="%1.1f%%")
plt.title("pie plot")
plt.show()


import pandas as pd
data=pd.read_csv(r"C:\Users\Soumitra\Desktop\machine learning and pyfiles\Datasets\Churn_Modelling.csv")

#statistical analysis
out=data.describe()

#drop irrelavant items
data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
data.Exited[data.Exited==1].count()
data.Age[data.Exited==1].mean()
data.Age[data.Exited==1].median()
data.Age[data.Exited==1].mode()
data.Age[data.Exited==1].var()
data.Age[data.Exited==1].std()
data.Age.mean()
data.groupby(["Geography","Gender"])["EstimatedSalary"].mean()


import seaborn as sns
plt.figure(figsize=(12,5))
sns.distplot(data.CreditScore[data.Exited==1])
sns.distplot(data.CreditScore[data.Exited==0])
plt.legend(["leaving","Not Leaving"])
plt.show()

plt.figure(figsize=(12,5))
sns.distplot(data.Age[data.Exited==1])
sns.distplot(data.Age[data.Exited==0])
plt.legend(["leaving","Not Leaving"])
plt.show()

sns.countplot(data.Geography)
plt.show()
sns.countplot(data.Geography[data.Exited==1])
plt.show()

sns.countplot(data.Gender[data.Exited==1])

plt.figure(figsize=(12,5))
sns.swarmplot(x=data.Geography,y=data.CreditScore,hue=data.Exited)
plt.show()

plt.figure(figsize=(12,5))
sns.swarmplot(x=data.Geography,y=data.Age,hue=data.Exited)
plt.show()

cor=data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()

