from os.path import dirname, abspath
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mplimg
from sklearn.utils import shuffle
from contextlib import redirect_stdout
import os
from os.path import dirname, abspath
import torch

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten, Conv2D, Activation
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras


homedir = dirname(dirname(abspath(__file__)))
model_dir = homedir + '/output/CNN_from_scratch/CNN1/2 epochs/'
test_videos_dir = homedir + '/test_videos/'
test_videos_table_file = test_videos_dir + 'faces_per_video_list.txt'

train_images = []

for i, file in enumerate(os.listdir(test_videos_dir + 'no_avg/')):
    if file.endswith(".pt"):
        tlist = torch.load(test_videos_dir + 'no_avg/' + file)
        for t in tlist:
            train_images.append(t.permute(1, 2, 0).int().numpy())

with open(test_videos_table_file, 'r') as f:
    test_videos_table = [line.split() for line in f]

print('data loaded')

for file in os.listdir(model_dir):
    if file.endswith(".h5"):
        model_file = model_dir + file
try:
    model = keras.models.load_model(model_file)
except:
    print('the model file didn\'t work')
    exit(-1)

train_images = np.array(train_images, dtype=float)
print('predicting')
predictions = model.predict(train_images)

i = 0
video_predictions = []
for video, num_faces in test_videos_table:
    mean_score = np.mean(predictions[i:int(num_faces)])
    video_predictions.append((video, int(mean_score > 0.5)))
    i += int(num_faces)

with open(test_videos_dir + 'video_predictions.txt', 'w+') as f:
    f.write('\n'.join('%s %s' % x for x in video_predictions))

