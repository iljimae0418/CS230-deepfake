import os
from os.path import dirname, abspath
import sys, time
import ntpath
import json
import cv2
import numpy as np
import pandas as pd
from IPython.display import Video
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

n_frames = 20
resize = 0.5

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    img = np.array(img, dtype='uint8')
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

homedir = dirname(dirname(abspath(__file__)))
train_dir = homedir + '/train_sample_videos/'
output_dir = homedir + '/output2'

json_file = []
for file in os.listdir(train_dir):
    if file.endswith(".json"):
        json_file = train_dir + file

if len(json_file) == 0:
    print("No json file", file=sys.stderr)
    exit()

'''
with open(json_file) as f:
    data = json.load(f)
'''

data = pd.read_json(json_file)
data = data.transpose()
data.reset_index(inplace=True)
data.index = range(data.shape[0])

train_video_files = []
train_labels = []

for index, row in data.iterrows():
    if not(row['label'] == 'FAKE' or row['label'] == 'REAL') or len(row['index']) == 0:
        continue
    if row['label'] == 'FAKE':
        train_labels.append(0)
    elif row['label'] == 'REAL':
        train_labels.append(1)
    train_video_files.append(train_dir + row['index'])

#train_video_files = [train_dir + x for x in os.listdir(train_dir)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = MTCNN(margin=50, keep_all=True, post_process=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device)


frames = []
faces = []
average_faces = []
#framed_train_labels = []
timestart = time.time()
'''
if not(os.path.exists(output_dir + '/face_tensors/')):
    os.makedirs(output_dir + '/face_tensors/')

if not(os.path.exists(output_dir + '/avg_face_tensors/')):
    os.makedirs(output_dir + '/avg_face_tensors/')

if not(os.path.exists(output_dir + '/frames_no_face_detected/')):
    os.makedirs(output_dir + '/frames_no_face_detected/')
empty_frames = output_dir + '/frames_no_face_detected/'

if not(os.path.exists(output_dir + '/orig_faces/')):
    os.makedirs(output_dir + '/orig_faces/')

# np.savetxt(output_dir + '/train_labels.txt', np.array(train_labels), fmt='%4d')

if not(os.path.exists(output_dir + '/avg_orig_faces/')):
    os.makedirs(output_dir + '/avg_orig_faces/')
'''
faces_labels = []
avg_labels_table = []

for i, (video, label) in enumerate(zip(train_video_files, train_labels)):
    num_faces = 0
    v_cap = cv2.VideoCapture(video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    video_name, _ = os.path.splitext(os.path.basename(video))
    for j in range(v_len):
        v_cap.grab()
        if j in sample:
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize([int(d * resize) for d in frame.size])
            # frames.append(frame)

            face = detector(frame)

            if type(face) != type(None):
                faces.extend(face)
                for k, f in enumerate(face):
                    #framed_train_labels.append(label)
                    faces_labels.append(label)
                    if j == 0 and k == 0:
                        sum_faces = f
                    else:
                        sum_faces += f
                    num_faces += 1

                    #f = f.permute(1, 2, 0).int().numpy()
                    #plt.imshow(f)
                    #plt.savefig(output_dir + '/orig_faces/' + str(j) + '_' + str(k) + video_name)
                    #labels_table.append((str(j) + '_' + str(k) + '_' + video_name, label))
            '''
            else:
                video_name, _ = os.path.splitext(os.path.basename(video))
                plt.imshow(frame)
                plt.savefig(empty_frames + video_name + '_frame_' + str(j) + 'of' + str(v_len))
                
                plt.imshow(frame)
                plt.show()                
                fig, axes = plt.subplots(1, len(face))
                axes.imshow(face[0].permute(1, 2, 0).int().numpy())
                axes.axis('off')
                '''
    avg_face = torch.div(sum_faces, num_faces)
    average_faces.append(avg_face)


    v_cap.release()


    #if i + 1 % 50 == 0:
    print(str(i + 1) + ' videos processed')
    '''
        v_range = str(i - 100) + '_' + str(i)
        torch.save(faces, output_dir + '/face_tensors/tensors_' + v_range + '.pt')
        np.savetxt(output_dir + '/frame_labels' + v_range + '.txt', np.array(framed_train_labels), fmt='%4d')
        torch.save(average_faces, output_dir + '/avg_face_tensors/avg_tensors_' + v_range + '.pt')
        faces = []
        average_faces = []
        framed_train_labels = []
    '''

np.savetxt(output_dir + '/faces_labels.txt', faces_labels, fmt='%4d')
np.savetxt(output_dir + '/avg_faces_labels.txt', train_labels, fmt='%4d')

torch.save(faces, output_dir + '/face_tensors.pt')
torch.save(average_faces, output_dir + '/avg_face_tensors.pt')
#np.savetxt(output_dir + '/frame_labels_to_' + str(len(train_video_files)) + '.txt', np.array(framed_train_labels), fmt='%4d')

timeend = time.time()
print(timeend - timestart)
