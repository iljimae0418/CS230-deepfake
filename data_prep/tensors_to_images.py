import torch
import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import numpy as np

homedir = dirname(dirname(abspath(__file__)))
output_dir = homedir + '/output'
all_tensors_dir = output_dir + '/no_averaging/face_tensors/'
avg_tensors_dir = output_dir + '/averaging/avg_face_tensors/'

all_tensors = []
avg_tensors = []
images = []

if not(os.path.exists(output_dir + '/averaging/images/')):
    os.makedirs(output_dir + '/averaging/images/')

if not(os.path.exists(output_dir + '/no_averaging/images/')):
    os.makedirs(output_dir + '/no_averaging/images/')

for file in os.listdir(all_tensors_dir):
    if file.endswith(".pt"):
        tlist = torch.load(all_tensors_dir + file)
        for i, t in enumerate(tlist):
            plt.imshow(t.permute(1, 2, 0).int().numpy())
            plt.savefig(output_dir + '/no_averaging/images/' + str(i))

for file in os.listdir(avg_tensors_dir):
    if file.endswith(".pt"):
        tlist = torch.load(all_tensors_dir + file)
        for i, t in enumerate(tlist):
            plt.imshow(t.permute(1, 2, 0).int().numpy())
            plt.savefig(output_dir + '/averaging/images/' + str(i))
