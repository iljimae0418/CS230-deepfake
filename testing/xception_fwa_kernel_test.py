''' labels are FAKE = 1, REAL = 0'''
import numpy as np 
import pandas as pd 
import os
import cv2 
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
from PIL import Image 
import tensorflow as tf
import tensorflow.keras
import os
from tensorflow.keras import models,layers
from tensorflow.keras.models import Model,model_from_json,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torchvision


os.system('pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl')
from facenet_pytorch import MTCNN 

model_path = '../input/frame-by-frame-xception1/model00000070.h5'
model = tf.keras.models.load_model(model_path) 

# function to extract all frames from the video
# n only saves the nth frame. If n == -1 saves every frame
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

# normalizes image data 
def processSingleData(image): 
    image = image.reshape(-1,244,244,3) 
    image = image.astype('float64')
    image = image/255.0 
    return image 

'''
The algorithm works as follows: 
If the number fake probabilites (i.e. > 0.5) is at least the number of threshold, we take the average of the top 20% highest predictions. 
If not,we take the average of top 20% lowest predictions.  
The threshold is a hyperparameter. 
'''
def makePredictions(video_paths,mtcnn): 
    ret = [] 
    trial = 1
    for path in video_paths:  
        print("Trial = {}........".format(trial)) 
        frames = getFrames(path,5) # get every 5th frame 
        threshold = int(len(frames)*0.4) # hyperparameter   
        avgCnt = int(len(frames)*0.2) # hyperparameter 
        predictions = []  
        for frame in frames: 
            faces = mtcnn(frame)
            if faces is None: # MTCNN failed to detect faces or there is no face 
                predictions.append(0.5) 
                continue 
            for face in faces: 
                face = face.permute(1,2,0).int().numpy() 
                face_img = processSingleData(face) 
                single_prediction = model.predict_on_batch(face_img)[0]  
                predictions.append(single_prediction)
        predictions.sort(reverse=True) 
        cnt = 0 # cnt how many predictions are fake.  
        for p in predictions: 
            if p >= 0.5:  
                cnt = cnt+1 
        final_prediction = 0 
        if cnt >= threshold: # video is fake, so take the average of the top 20% largest predictions
            final_prediction = np.mean(predictions[0:avgCnt]) 
        else: # video is real, so take the average of the top 20% smallest predictions 
            final_prediction = np.mean(predictions[len(predictions)-avgCnt:])
        #print(final_prediction)
        ret.append(final_prediction)
        trial = trial+1 
    return ret 
    
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/' 
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
test_video_files.sort() # submission is in alphabetical order I believe  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=244, margin=100, thresholds=[0.7, 0.95, 0.85], keep_all=True,post_process=False, device=device)

predictions = makePredictions(test_video_files,mtcnn)
ss = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')
ss['label'] = predictions 
ss.to_csv('submission.csv',index=False)  
ss.head() 


