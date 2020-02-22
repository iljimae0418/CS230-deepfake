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

# collect data from train directory
train_dir = './faces_train/'
train_faces = [train_dir + x for x in os.listdir(train_dir)]
train_faces.sort()

# collect data from test directory
test_dir = './faces_test/'
test_faces = [test_dir + x for x in os.listdir(test_dir)]
test_faces.sort()
test_faces = test_faces[1:]

dim = (64,64)

# resize test_faces to 64x64
for path in test_faces:
    image = cv.imread(path)
    image = cv.resize(image,dim)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    query = re.search('./faces_test/(.*)',path)
    pic_name = query.group(1)
    save_path = "./face_test_64/"+pic_name
    cv.imwrite(save_path,cv.cvtColor(image,cv.COLOR_BGR2RGB))

# resize train_faces to 64x64
for path in train_faces:
    image = cv.imread(path)
    image = cv.resize(image,dim)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    query = re.search('./faces_train/(.*)',path)
    pic_name = query.group(1)
    save_path = "./face_train_64/"+pic_name
    cv.imwrite(save_path,cv.cvtColor(image,cv.COLOR_BGR2RGB))
