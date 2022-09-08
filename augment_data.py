from itertools import cycle
import numpy as np
import sys

sys.path.append('../test')

from utils.plot_lines import animation_plot

import os
import h5py
import utils.common

root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, "data")
type = '_ELMD'  # '' or '4DCVAEGCN' or '_ELMD'
feature_file_name = 'features' + type + '.h5'
label_file_name = 'labels' + type + '.h5'
npy_file_name = 'features' + type + '.npy'

npy_file_name_augmented = 'features' + type + 'aug.npy'
npy_label_augmented = 'labels' + type + 'aug.npy'

fl = h5py.File(os.path.join(data_path, label_file_name), 'r')
num_samples = len(fl.keys())
label = np.empty(num_samples)

for sidx in range(num_samples):
  label[sidx] = fl[list(fl.keys())[sidx]][()]

# file_path = os.path.join(data_path, feature_file_name)
npy_path = os.path.join(data_path, npy_file_name)
npy_path_aug = os.path.join(data_path, npy_file_name_augmented)
npy_path_label_aug = os.path.join(data_path, npy_label_augmented)

data = np.load(npy_path)
# data = np.load("test.npy")
clips = data

print("Orig shape: ", clips.shape)

clips = np.swapaxes(clips, 1, 2)
count = 0

new_data = []
new_label = []
for n in label:
  if n == 3:
    new_label.extend([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    new_data.append(data[count])
    new_data.append(data[count] + 0.2)
    new_data.append(data[count] * 0.5)
    new_data.append(data[count] * 1.2)
    new_data.append(data[count] * 1.5)
    new_data.append(data[count] + 0.8)
    new_data.append(data[count] + 0.5)
    new_data.append(data[count] + 1)
    new_data.append(data[count] * -1.3)
    new_data.append(data[count] - 0.8)
    new_data.append(data[count] - 0.1)
  elif n == 2:
    new_label.extend([2, 2, 2, 2])
    new_data.append(data[count])
    new_data.append(data[count] + 0.2)
    new_data.append(data[count] * 0.5)
    new_data.append(data[count] * 1.2)
  elif n == 1:
    new_label.extend([1, 1])
    new_data.append(data[count])
    new_data.append(data[count] + 0.2)
  elif n == 0:
    new_label.extend([0])
    new_data.append(data[count])
  count += 1
new_data = np.array(new_data)
new_label = np.array(new_label)
print(new_data.shape, new_label.shape)
np.save(npy_path_aug, new_data)
np.save(npy_path_label_aug, new_label)

print("After swap: ", clips.shape)
sad = clips[69:70]  # predicted Neutral
neutral = clips[1830:1831]  # Predicted Neutral
happy = clips[1798:1899]  # Happy
angry = clips[1698:1699]  # angry
# seq1 = clips[69:70]
# seq2 = clips[1:2]
# seq3 = clips[2:3]
# temp = np.swapaxes(angry, 1, 2)
# print(temp.shape)
# print(temp[0][1])
# temp = temp + 0.2
# print(temp[0][1])

# animation_plot([angry, angry * 2], filename="angry", interval=100.15, predicted="angry")
