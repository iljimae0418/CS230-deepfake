import os
from os.path import dirname, abspath
import sys, time
import ntpath
import json
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
import csv
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch


n_stack_frames = 40
n_independent_frames = 5
face_shape = (244, 244, 3)

homedir = dirname(dirname(abspath(__file__)))
train_dir = homedir + '/data/train_parts_05/'
output_dir = homedir + '/output/'
output_individual = output_dir + "individual_frames/"
output_stacked = output_dir + "stacked_frames/"
audio_fakes_table = homedir + '/metadata_audio_altered.csv'

json_files = []

for subdir, dirs, files in os.walk(train_dir):
    for dir in dirs:
        for file in os.listdir(train_dir + dir):
            if file.endswith(".json"):
                json_files.append(train_dir + dir + '/' + file)

if len(json_files) == 0:
    print("No json files", file=sys.stderr)
    exit()

data = pd. DataFrame()
for file in json_files:
    file_data = pd.read_json(file)
    file_data = file_data.transpose()
    file_data.reset_index(inplace=True)
    file_data.index = range(data.shape[0], data.shape[0] + file_data.shape[0])
    path = os.path.dirname(file)
    file_data['folder'] = os.path.basename(path)
    data = data.append(file_data)

train_video_files = []
table_to_save = []
train_labels = []

data.sort_values(by=['index'], inplace=True)

audio_fakes_data = pd.read_csv(audio_fakes_table)
audio_fakes_data.reset_index(inplace=True)
audio_fakes_data.index = range(audio_fakes_data.shape[0])

init_data_size = data.shape[0]

data = data[~data['index'].isin(audio_fakes_data['train_mp4'])][:10]

for index, row in data.iterrows():
    if not (row['label'] == 'FAKE' or row['label'] == 'REAL') or len(row['index']) == 0:
        continue
    if row['label'] == 'FAKE':
        train_labels.append(1)
    elif row['label'] == 'REAL':
        train_labels.append(0)
    train_video_files.append(train_dir + row['folder'] + '/' + row['index'])
    table_to_save.append([row['folder'], row['index']])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(margin=100, image_size=244, keep_all=True, thresholds=[0.7, 0.95, 0.85], post_process=False, device=device)

if not(os.path.exists(output_individual)):
    os.makedirs(output_individual)
if not(os.path.exists(output_stacked)):
    os.makedirs(output_stacked)

print("processing videos")

individual_frames_dict = {}
stacked_frames_dict = {}
stacked_frames_df = pd.DataFrame(columns=['index', 'fake', 'num_empty_frames'])

starttime = time.time()

for i, (video, label) in enumerate(zip(train_video_files, train_labels)):
    num_faces = 0
    v_cap = cv2.VideoCapture(video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_stack = np.linspace(0, v_len - 1, n_stack_frames).astype(int)
    sample_independent = np.linspace(0, v_len - 1, n_independent_frames).astype(int)
    video_name = os.path.basename(video)

    one_faced_video = True
    stacked_faces = []
    num_empty_stacked_faces = 0

    for j in range(v_len):
        if label == 1 and not one_faced_video:
            break
        v_cap.grab()
        if label == 0:
            if j in sample_independent:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                faces = detector(frame)

                if faces is not None:
                    for k, face in enumerate(faces):
                        np.save(output_individual + f"{video_name}_{j}_{k}", face.permute(1, 2, 0).int().numpy())
                        individual_frames_dict[f"{video_name}_{j}_{k}.npy"] = label

        if j in sample_stack and one_faced_video:
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            faces = detector(frame)

            if faces is None:
                stacked_faces.append(np.zeros(face_shape))
                num_empty_stacked_faces += 1
                continue
            if len(faces) > 1:
                one_faced_video = False
                continue
            stacked_faces.append(faces[0].permute(1, 2, 0).int().numpy())

    v_cap.release()

    if one_faced_video and len(stacked_faces) == n_stack_frames:
        stacked_faces = np.array(stacked_faces)
        np.save(output_stacked + video_name, stacked_faces)
        #stacked_frames_dict[video_name] = label
        stacked_frames_df = stacked_frames_df.append(
            [{'index':video_name,'fake':label,'num_empty_frames':num_empty_stacked_faces}], ignore_index=True)

    if (i + 1) % 5 == 0:
        currtime = time.time()
        time_dif = currtime - starttime

        print(str(i + 1) + ' videos processed in ' + str(time_dif/60) + " min")

with open(output_individual + 'datatable.csv', 'w+') as f:
    w = csv.writer(f)
    w.writerows(individual_frames_dict.items())
'''
with open(output_stacked + 'datatable.csv', 'w+') as f:
    w = csv.writer(f)
    w.writerows(stacked_frames_dict.items())
'''

stacked_frames_df.to_csv(output_stacked + 'dataframe.csv')

print("total num of individual frames: " + str(len(individual_frames_dict)))
print("total num of stacks: " + str(stacked_frames_df.shape[0]))
print("total num of whole stacks: " + str(stacked_frames_df.loc[stacked_frames_df['num_empty_frames'] == 0].shape[0]))
