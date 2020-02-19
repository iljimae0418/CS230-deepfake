import cv2 as cv
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from IPython.display import Video
from IPython.display import HTML
import face_recognition
from tqdm import tqdm
import pickle
from PIL import Image
import PIL
import re

# get test videos from the test video file directory
test_dir = './test_videos/'
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
test_video_files.sort()

def DEBUG(list):
    for i in range(0,len(list)):
        print(list[i])

def getFrames(video_file):
    frames = []
    cap = cv.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    return frames

def pullFaces(frames):
    for i in range(0,len(frames),25):
        frame = frames[i]
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            print("Face detected in frame %d" % i)
            # some face is found
            # fetch the first face
            top,right,bottom,left = face_locations[0]
            face = frame[top:bottom,left:right]
            image = cv.cvtColor(face,cv.COLOR_BGR2RGB)
            return image
    print("No Face Detected!!!")
    return

def printImage(image):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    plt.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image)
    plt.show()

#DEBUG(test_video_files)

#cannot detect faces in some videos. Manually obtained faces then resized.
undetected = []
dim = (224,224) # to match the size of the train images
for i in range(0,len(test_video_files)):
    print("Going through video %d ........" % i)
    test_video = test_video_files[i]
    frames = getFrames(test_video)
    image = pullFaces(frames)
    if image is None:
        query = re.search('./test_videos/(.*).mp4',test_video)
        pic_name = query.group(1)+".jpg"
        undetected.append(pic_name)
        continue
    image = cv.resize(image,dim)
    query = re.search('./test_videos/(.*).mp4',test_video)
    pic_name = query.group(1)+".jpg"
    save_path = "./faces_test/"+pic_name
    cv.imwrite(save_path,cv.cvtColor(image,cv.COLOR_BGR2RGB))

for x in undetected:
    print(x)
