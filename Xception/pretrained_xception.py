import cv2 as cv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video
from IPython.display import HTML
import face_recognition
from tqdm import tqdm
import pickle
from PIL import Image
import PIL
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import h5py
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.xception import decode_predictions
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

def DEBUG(X,Y):
    print(X.shape)
    print(Y.shape)

def printImage(image):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    plt.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image)
    plt.show()

# images adopted from kaggler's data
data = pd.read_csv('./metadata.csv')
names = []
train_labels = []
fake_cnt = 0
real_cnt = 0
for idx in data.index:
    if fake_cnt == 1000 and real_cnt == 500:
        break
    if data['label'][idx] == "FAKE" and fake_cnt < 1000:
        names.append(data['videoname'][idx][:-4])
        train_labels.append(1)
        fake_cnt = fake_cnt+1
    elif data['label'][idx] == "REAL" and real_cnt < 500:
        names.append(data['videoname'][idx][:-4])
        train_labels.append(0)
        real_cnt = real_cnt+1
train_labels = np.asarray(train_labels)

for i in range(10):
    print("name = {}, label = {}".format(names[i],train_labels[i]))

# setting up train data
train_dir = './faces_train/'
train_img = []
for i in range(len(names)):
    path = train_dir + names[i] + ".jpg"
    image = cv.imread(path)
    image = cv.resize(image,(244,244))
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    train_img.append(image)

train_img = np.asarray(train_img)
train_img = train_img.reshape(-1,244,244,3)
train_img = train_img.astype('float64')
train_img = train_img/255.0

def xception(percent2retrain,dimension):
    xception_model = Xception(input_shape=dimension,weights='imagenet',include_top=False)
    # freeze base layers
    if percent2retrain < 1:
        for layer in xception_model.layers[:-int(len(xception_model.layers)*percent2retrain)]: layer.trainable = False

    model = Sequential()
    model.add(xception_model)
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

model = xception(1,(244,244,3))
print(model.summary())
model.fit(train_img,train_labels,epochs=5)
model.save('pretrained_xception_prototype2.h5')
