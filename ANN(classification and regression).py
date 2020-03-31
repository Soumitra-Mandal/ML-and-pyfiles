# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:11:43 2019

@author: Lenovo
"""
"""
ARTIFICIAL NEURAL NETWORK
cross entropy=-summation[yln(p)+(1-y)(ln(1-p))]
feed forward
back propagation
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
n_pts=500
x=np.linspace(-3,3,n_pts)
y=(5*x+3)+np.random.uniform(-5,5,n_pts)
plt.scatter(x,y)
plt.show()

model=Sequential()
model.add(Dense(50,input_dim=1,activation='relu'))

model.add(Dense(1))
model.compile(Adam(lr=0.01),loss='mse')

h=model.fit(x,y,verbose=1,epochs=50,validation_split=0.1)

plt.plot(h.history["loss"])
plt.show()

pred=model.predict(x)
plt.scatter(x,y,color="red")
plt.plot(x,pred,color='yellow')
plt.show()

"""
non-linear 
"""
np.random.seed(0)
n_pts=500
x=np.linspace(-3,3,n_pts)
y=np.sin(x)+np.random.uniform(-0.5,0.5,n_pts)
plt.scatter(x,y,color='b')
plt.show()

model=Sequential()
model.add(Dense(50,input_dim=1,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(1))
model.compile(Adam(lr=0.01),loss='mse')
h=model.fit(x,y,verbose=1,epochs=50)


plt.plot(h.history["loss"])
plt.show()

pred=model.predict(x)
plt.scatter(x,y,color="red")
plt.plot(x,pred,color='black')
plt.show()


"""
classification problem using keras
"""
import matplotlib.pyplot as plt
n_pts=500
np.random.seed(0)

Xa=np.array([np.random.normal(13,2,n_pts),np.random.normal(13,2,n_pts)]).T
Xb=np.array([np.random.normal(8,2,n_pts),np.random.normal(8,2,n_pts)]).T

X=np.vstack((Xa,Xb))
Y=np.matrix(np.append(np.zeros(n_pts),np.ones(n_pts))).T
plt.scatter(X[:n_pts,0],X[:n_pts,1],color="r")
plt.scatter(X[n_pts:,0],X[n_pts:,1],color="b")
plt.show()

#importing keras functions

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
model=Sequential()
model.add(Dense(units=1,input_shape=(2,),activation="sigmoid"))
adam=Adam(lr=0.1)
model.compile(adam,loss="binary_crossentropy",metrics=["accuracy"])
history=model.fit(x=X,y=Y,verbose=1,batch_size=50,validation_split=0.1,epochs=10)
plt.plot(history.history["loss"],color="c")
plt.plot(history.history["val_loss"])
plt.legend(["training","validation"])
plt.title("Loss")
plt.xlabel("epoch")

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["training","validation"])
plt.title("Accuracy")
plt.xlabel("epoch")


def plot_decision_boundary(X,y,model):
    x_span = np.linspace(min(X[:,0])-1,max(X[:,0])+1)
    y_span =  np.linspace(min(X[:,1])-1,max(X[:,1])+1)
    
    xx,yy=np.meshgrid(x_span,y_span)
    
    xx_,yy_=xx.ravel(),yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx,yy,z)
plot_decision_boundary(X,Y,model)
plt.scatter(X[:n_pts,0],X[:n_pts,1])
plt.scatter(X[n_pts:,0],X[n_pts:,1])

x=7.5
y=10

point=np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x],[y],marker='o',markersize=10,color="black")
print(prediction)

"""
classification using keras problem 2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
n=500
X,Y = datasets.make_circles(n_samples=n, random_state=0,noise=0.1,factor=0.2)
print(X)
print(Y)

plt.scatter(X[Y==0,0],X[Y==0,1])
plt.scatter(X[Y==1,0],X[Y==1,1])

model=Sequential()
model.add(Dense(4,input_shape=(2,),activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(Adam(lr=0.01),"binary_crossentropy",metrics=["accuracy"])
history=model.fit(x=X,y=Y,verbose=1,batch_size=20,validation_split=0.1,epochs=75,shuffle=True)
plt.plot(history.history["acc"])
plt.legend(["Accuracy"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.plot(history.history["loss"])
plt.legend(["loss"])
plt.title("loss")
plt.xlabel("epoch")

def plot_decision_boundary(X,Y,model):
     x_span = np.linspace(min(X[:,0])-0.25,max(X[:,0])+0.25)
     y_span =  np.linspace(min(X[:,1])-0.25,max(X[:,1])+0.25)
     
     xx,yy=np.meshgrid(x_span,y_span)
     grid=np.c_[xx.ravel(),yy.ravel()]
     pred_func = model.predict(grid)
     z = pred_func.reshape(xx.shape)
     plt.contourf(xx,yy,z)
plot_decision_boundary(X,Y,model)
plt.scatter(X[Y==0,0],X[Y==0,1])
plt.scatter(X[Y==1,0],X[Y==1,1])

a=0
b=0.75

point=np.array([[a,b]])
prediction = model.predict(point)
plt.plot([a],[b],marker='o',markersize=10,color="black")
print(prediction)


"""
churn modelling dataset claasificatiion using keras
"""
import pandas as pd
data=pd.read_csv(r"C:\Users\Lenovo\Desktop\ai and ml dec\data\Churn_Modelling.csv")
data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
data.dtypes


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data.Geography=le.fit_transform(data.Geography)
data.Gender=le.fit_transform(data.Gender)
ip=data.drop(["Exited"],axis=1)
op=data["Exited"]

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(categorical_features=[1])
ip=ohe.fit_transform(ip).toarray()

from sklearn.model_selection import train_test_split 
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
sc.fit(xtr)
xtr=sc.transform(xtr)
xts=sc.transform(xts)

model=Sequential()
model.add(Dense(10,input_shape=(12,),activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(Adam(lr=0.01),"binary_crossentropy",metrics=["accuracy"])
history=model.fit(x=xtr,y=ytr,verbose=1,batch_size=50,validation_split=0.1,epochs=10)
plt.plot(history.history["acc"])
plt.legend(["Accuracy"])
plt.title("Accuracy")
plt.xlabel("epoch")


plt.plot(history.history["loss"])
plt.legend(["loss"])
plt.title("loss")
plt.xlabel("epoch")

