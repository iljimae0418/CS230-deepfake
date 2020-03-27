import pandas as pd
import numpy as np
import json
import cv2
from mtcnn import MTCNN
from facenet_pytorch import MTCNN
import torch
import torchvision
from PIL import Image
import PIL
import time
import matplotlib.pyplot as plt
import os
from random import sample


start_time = time.time()

metadata = pd.read_json('./metadata.json').T # read from json file
metadata['name'] = metadata.index

X_dat = [] # face images (244,244,3)
Y_dat = [] # labels

filepath = './dfdc_train_part_2/' # name of directory where the file is stored

# function to extract all frames from the video
# n only saves the nth frame. If n == -1 saves every frame
# on average, a video has about 300 frames, so if n = 5, every 5th frame will be saved, so in total 60 frames
def getFrames(video_file,n):
    frames = []
    cap = cv2.VideoCapture(video_file)
    index = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if n == -1:
                frames.append(frame)
            elif n != -1 and index%n == 0:
                frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            index = index+1
        else:
            break
    cap.release()
    return frames

# to actually see the images themselves
def printImage(image):
    plt.imshow(image)
    plt.show()

# checks if the video contains single faces
def singleFaceCheck(frames,mtcnn):
    num_faces = -1
    for i in range(len(frames)):
        faces = mtcnn(frames[i])
        if faces is None:
            continue
        if len(faces) == 1:
            num_faces = 1
            break
        elif len(faces) >= 2:
            num_faces = 2
            break
    return num_faces==1

# alternative way of checking if two images are the same
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())


# hyperparameters
n = 5 # extract every nth frame
m = 6 # choose m random frames from the extracted frames

# for some stats
real_cnt = 0
fake_cnt = 0
real_vids = 0
fake_vids = 0
single_face_vids = 0

# face extraction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=244, margin=100, thresholds=[0.7, 0.95, 0.85], keep_all=True,post_process=False, device=device)
idx = 1 # for testing

# iterate over the videos
for index,row in metadata.iterrows():
    print("Processing video {}........".format(idx))
    frames = getFrames(filepath+row['name'],n) # get every nth frame
    assert (m <= len(frames)),"We cannot choose more than the number of frames extracted!"
    if singleFaceCheck(frames,mtcnn) == True:
        if row['label'] == 'REAL': # if video is real
            for x in sample(range(0,len(frames)),m): # pick m random frames, m <= total number of extracted frames
                faces = mtcnn(frames[x])
                face = faces[0].permute(1,2,0).int().numpy()
                # printImage(face)
                X_dat.append(face)
                Y_dat.append(0)
                real_cnt = real_cnt+1
            real_vids = real_vids + 1
        elif row['label'] == 'FAKE': # if video is fake
            frames_original = getFrames(filepath+row['original'],n)
            ub = min(len(frames),len(frames_original))
            for x in sample(range(0,ub),m):
                if not np.all(frames[x] == frames_original[x]): # if the captured deepfake and original frames are not the same, it is indeed deepfake
                    faces = mtcnn(frames[x])
                    if faces is None:
                        continue
                    face = faces[0].permute(1,2,0).int().numpy()
                    # printImage(face)
                    X_dat.append(face)
                    Y_dat.append(1)
                    fake_cnt = fake_cnt+1
            fake_vids = fake_vids+1
        single_face_vids = single_face_vids+1
    idx = idx+1


X_dat = np.asarray(X_dat)
Y_dat = np.asarray(Y_dat)

# normalize
X_dat = X_dat/255.0
# save data as numpy
save_X = 'chunk2_X'
save_Y = 'chunk2_Y'
np.save(save_X,X_dat)
np.save(save_Y,Y_dat)

# print stats
print("============================= STATISTICS =============================")
print("Train X Shape = {}".format(X_dat.shape))
print("Train Y Shape = {}".format(Y_dat.shape))
print("Total number of videos = {}".format(len(metadata)))
print("Total number of real videos = {}".format(real_vids))
print("Total number of fake videos = {}".format(fake_vids))
print("Total number of single face videos = {}".format(single_face_vids))
print("Total number of real captured faces = {}".format(real_cnt))
print("Total number of fake captured faces = {}".format(fake_cnt))
print("Time taken for the program = {} seconds".format(time.time()-start_time))
