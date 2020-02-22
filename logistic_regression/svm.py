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
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split

# collect data from train directory
train_dir = './face_train_64/'
train_faces = [train_dir + x for x in os.listdir(train_dir)]
train_faces.sort()

# obtain images from the path names
train_set_x_orig = []
train_face_subset = train_faces[:1500]
for path in train_face_subset:
    image = cv.imread(path)
    train_set_x_orig.append(image)


data = pd.read_csv("metadata.csv")
data.head()

# obtain y_train
# y is 1 if the video is FAKE, 0 if REAL
y_train = data['label'].to_numpy()
y_train = y_train[:1500]
for i in range(len(y_train)):
    if y_train[i] == "FAKE":
        y_train[i] = 1
    else:
        y_train[i] = 0

# train/validation set split using stratitification 
train_set_x_orig, validation_set_x_orig, y_train, y_validation = train_test_split(train_set_x_orig,y_train,stratify=y_train,test_size=0.2,random_state=42)


train_set_x_orig = np.asarray(train_set_x_orig)
validation_set_x_orig = np.asarray(validation_set_x_orig)

train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
validation_set_x_orig = validation_set_x_orig.reshape(validation_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_orig/255
validation_set_x = validation_set_x_orig/255


clf = svm.SVC(kernel="linear",probability=True,random_state=42)
y_train = y_train.astype('int')
clf.fit(train_set_x.T,y_train)
y_pred = clf.predict(validation_set_x.T)
y_validation = y_validation.astype('int')
y_pred = y_pred.astype('int')
print("Validation Set Accuracy = ",metrics.accuracy_score(y_validation,y_pred))
print(confusion_matrix(y_validation,y_pred))
