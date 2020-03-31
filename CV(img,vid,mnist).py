# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:01:36 2019

@author: Soumitra
"""
"""
Computer Vision
conda install -c conda-forge opencv
"""
import cv2
img=cv2.imread(r"E:\wp\avengers-endgame-1920x1080-minimal-art-4k-18264.jpg")
cv2.imshow("lena",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)

#save image to folder
img=cv2.imread(r"C:\Users\Lenovo\Desktop\sample.jpg")
outpath=r"C:\Users\Lenovo\Desktop\img pro\sample.jpg"
cv2.imshow("lena",img)
cv2.imwrite(outpath,img)


print(img)
print(type(img))   
print(img.dtype)
print(img.shape)
print(img.ndim)
print(img.size)

#image display using matplotlib

import matplotlib.pyplot as plt
img=cv2.imread(r"E:\wp\avengers-endgame-1920x1080-minimal-art-4k-18264.jpg",1)
img=cv2.resize(img,(512,512))
plt.imshow(img)
plt.show()

#color change
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("RGB colormap")
plt.xticks([])
plt.yticks([])
plt.show()

"""
image of numbers using nmist
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
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

X_train=X_train/255
X_test=X_test/255

num_pixels = 784
X_train=X_train.reshape(X_train.shape[0],num_pixels)
X_test = X_test.reshape(X_test.shape[0],num_pixels)
def create_model():
    model=Sequential()
    model.add(Dense(10,input_dim=num_pixels,activation="relu"))
    model.add(Dense(30,activation="relu"))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(num_classes,activation="softmax"))
    model.compile(Adam(lr=0.01),"categorical_crossentropy",metrics=["accuracy"])
    return model

model = create_model()
history=model.fit(X_train,Y_train,verbose=1,batch_size=200,validation_split=0.1,epochs=10,shuffle=1)

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
img=img.reshape(1,784)

prediction = model.predict_classes(img)
print("Predicted Digit:",str(prediction))

"""
canny edge detection algorithm
"""
import cv2
import numpy as np
image = cv2.imread(r"C:\Users\Lenovo\Desktop\sample.jpg")
canny = cv2.Canny(image,200,300)

cv2.imshow("canny",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Drawing Basic Geometric Shapes
"""

import numpy as np
import cv2

img = np.zeros((512,512,3),np.uint8)

cv2.line(img,(0,99),(99,0),(255,0,0),2)
cv2.rectangle(img,(40,60),(80,70),(200,255,200),2)
cv2.circle(img,(60,60),10,(0,0,255),2)
cv2.imshow("lena",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
face detection
"""
import cv2
import numpy as np

face_detection = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\Datasets\data\haarcascades\haarcascade_frontalface_default.xml")

img = cv2.imread(r"C:\Users\Soumitra\Desktop\sample.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
cv2.imshow("Face_detect",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
video detection
"""
import cv2
cap=cv2.VideoCapture(r"D:\dc++\Phineas And Ferb\Season 01\Phineas and Ferb.S01E10.The Magnificent Few.720p.x265.HEVC.MaGiC.mkv")#0 for webcam

if cap.isOpened():
    ret,frame=cap.read()
else:
    ret=False

#loop for video
    while ret:
        ret,frame=cap.read()
        cv2.imshow("lena",frame)
        if cv2.waitKey(1)==27:#esc key breaks the loop
            break
cv2.destroyAllWindows()


