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

# collect data from train directory
train_dir = './face_train_64/'
train_faces = [train_dir + x for x in os.listdir(train_dir)]
train_faces.sort()
# collect data from test directory
'''
test_dir = './face_test_64/'
test_faces = [test_dir + x for x in os.listdir(test_dir)]
test_faces.sort()
'''

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

# train/validation set split using stratification
train_set_x_orig, validation_set_x_orig, y_train, y_validation = train_test_split(train_set_x_orig,y_train,stratify=y_train,test_size=0.2,random_state=42)

train_set_x_orig = np.asarray(train_set_x_orig)
validation_set_x_orig = np.asarray(validation_set_x_orig)

train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
validation_set_x_orig = validation_set_x_orig.reshape(validation_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_orig/255
validation_set_x = validation_set_x_orig/255

def sigmoid(z):
    z = z.astype(float)
    s = 1/(1+np.exp(-z))
    return s

# initialize weight and bias to zero
def initialize(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

# implement forward and backward propagation
def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1/m) * np.sum(np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1-A).T))
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    grads = {"dw":dw,
            "db":db}
    return grads, cost

def fit(w,b,X,Y,epochs,learning_rate,print_cost=False):
    costs = []
    for i in range(epochs):
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 3 == 0:
            costs.append(cost)
        if print_cost and i % 3 == 0:
            print("Cost after iteration %i: %f" % (i,cost))

    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}

    return params,grads,costs

def predict(w,b,X):
    m = X.shape[1]
    w = w.reshape(X.shape[0],1)
    # A[0][i] contains prediction probabilities
    A = sigmoid(np.dot(w.T,X)+b)
    return A

def score(x_train,y_train,x_val,y_val,epochs,learning_rate,print_cost=False):
    w,b = initialize(x_train.shape[0])
    parameters, grads, costs = fit(w, b, x_train, y_train, epochs, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    A_train = predict(w, b, x_train)
    A_val = predict(w, b, x_val)
    train_predictions = A_train[0]
    val_predictions = A_val[0]

    train_tobinary = np.zeros(len(train_predictions))
    val_tobinary = np.zeros(len(val_predictions))

    for i in range(0,len(train_predictions)):
        if train_predictions[i] > 0.5:
            train_tobinary[i] = 1
        else:
            train_tobinary[i] = 0

    for i in range(0,len(val_predictions)):
        if val_predictions[i] > 0.5:
            val_tobinary[i] = 1
        else:
            val_tobinary[i] = 0


    print("train accuracy: {} %".format(100-np.mean(np.abs(train_tobinary-y_train))*100))
    print("validation accuracy: {} %".format(100-np.mean(np.abs(val_tobinary-y_val))*100))



    info = {"costs":costs,
        "train_predictions":train_tobinary,
        "val_predictions":val_tobinary,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "epochs":epochs}
    return info



info = score(train_set_x, y_train, validation_set_x, y_validation, epochs = 255, learning_rate = 0.001, print_cost = True)

print(confusion_matrix(y_validation.astype('int32'),info["val_predictions"].astype('int32')))

plt.plot(info["costs"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()


'''
submission = pd.read_csv('sample_submission.csv')
df = pd.DataFrame({'filename':submission['filename'],
                    'label':predictions})
df.to_csv('simple_logistic_reg.csv', index=False)
'''
