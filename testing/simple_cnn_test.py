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
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import h5py

df = pd.read_json('./train_sample_videos/metadata.json').T
df = df.reset_index()

# get training data for y
train_labels = df['label'].to_numpy()
for i in range(0,len(train_labels)):
    if train_labels[i] == "FAKE":
        train_labels[i] = 1
    else:
        train_labels[i] = 0

train_labels = np.asarray(train_labels)


def getFirstFrame(video):
    vidcap = cv.VideoCapture(video)
    success, image = vidcap.read()
    dim = (200,200)
    if success:
        image = cv.resize(image,dim)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        #query = re.search('./train_sample_videos/(.*).mp4',video)
        #pic_name = query.group(1)+".jpg"
        #save_path = './train_sample_first_frames/'+pic_name
        #cv.imwrite(save_path,cv.cvtColor(image,cv.COLOR_BGR2RGB))

def printImage(image):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    plt.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image)
    plt.show()


# get first frames
train_dir = './train_sample_first_frames/'
train_firstframe = [train_dir + x for x in os.listdir(train_dir)]
train_firstframe.sort()

train_images = []
for path in train_firstframe:
    image = cv.imread(path)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    train_images.append(image)

train_images = np.asarray(train_images)
train_images = train_images.reshape(-1,200,200,3)
train_images = train_images/255

train_images = tf.convert_to_tensor(train_images,dtype=tf.float64)
train_labels = tf.convert_to_tensor(train_labels,dtype=tf.float64)


# build CNN
def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(10,(3,3),activation='relu',padding='same',input_shape=(200,200,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = create_model()
model.summary()
model.fit(train_images,train_labels,epochs=5)
model.save('bad_cnn.h5')

