# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:03:16 2019

@author: Soumitra
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
np.random.seed(0)
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape[0])

assert(X_train.shape[0]==Y_train.shape[0]),"The number of images is not equal to the number of labels."
assert(X_test.shape[0]==Y_test.shape[0]),"The number of images is not equal to the number of labels."
assert(X_train.shape[1:]==(28,28)),"The dimemsions of the images are not 28x28"
assert(X_test.shape[1:]==(28,28)),"The dimemsions of the images are not 28x28"

num_of_samples = []
cols= 5
num_classes=10

fig,axs=plt.subplots(nrows=num_classes,ncols=cols, figsize=(5,8))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        X_selected=X_train[Y_train == j]
        axs[j][i].imshow(X_selected[random.randint(0,len(X_selected-1)),:,:],cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i==2:
           axs[j][i].set_title(str(j))
           num_of_samples.append(len(X_selected))
           
print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes),num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)




X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_train=X_train/255
X_test=X_test/255

def lenet_model():
    model=Sequential()
    model.add(Conv2D(30,(5,5),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model=lenet_model()
history=model.fit(X_train,Y_train,verbose=1,epochs=10,shuffle=1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.title("loss")
plt.xlabel("epoch")

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["Accuracy","Val_accuracy"])
plt.title("Accuracy")
plt.xlabel("epoch")

score = model.evaluate(X_test,Y_test,verbose=0)
print(type(score))
print("Test score:",score[0])
print("Test accuracy:",score[1])


import requests
from PIL import Image


url="https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png"
response = requests.get(url,stream=True)
img=Image.open(response.raw)
plt.imshow(img,cmap=plt.get_cmap("gray"))



import cv2

img = np.asarray(img)
img = cv2.resize(img,(28,28))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.bitwise_not(img)
plt.imshow(img,cmap=plt.get_cmap("gray"))

img=img/255
img=img.reshape(1,28,28,1)

prediction = model.predict_classes(img)
print("Predicted Digit:",str(prediction))

layer1=Model(inputs=model.layers[0].input,outputs=model.layers[0].output)
layer1=Model(inputs=model.layers[0].input,outputs=model.layers[2].output)

visual_layer1,visual_layer2=layer1.predict(img),layer2.predict(img)
print(visual_layer1.shape)
print(visual_layer2.shape)

#layer1
plt.figure(figsize=(10,6))
for i in range(30):
    plt.subplot(6,5,i+1)
    plt.imshow(visual_layer1[0,:,:,i],cmap=plt.get_cmap('jet'))
    plt.axis("off")
    

#layer2
plt.figure(figsize=(10,6))
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(visual_layer2[0,:,:,i],cmap=plt.get_cmap('jet'))
    plt.axis("off")
    

"""
git init
git add . 
creaye repo cardata
"""