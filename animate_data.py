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
npy_file_name = 'features' + type + '.npy'

file_path = os.path.join(data_path, feature_file_name)
npy_path = os.path.join(data_path, npy_file_name)

# f = h5py.File(file_path, 'r')

# num_samples = len(f.keys())
# time_steps = 0
# cycles = 1
# joints = 16
# coords = 3
# data_list = []
# print("preparing data...")
# for sidx in range(num_samples):
#   f_group_key = list(f.keys())[sidx]
#   data_list.append(list(f[f_group_key]))  # Get the data
#   time_steps_curr = len(f[f_group_key])
#   # max steps
#   if time_steps_curr > time_steps:
#     time_steps = time_steps_curr
#   data = np.empty((num_samples, time_steps * cycles, joints * coords))

# for sidx in range(num_samples):
#   data_list_curr = np.tile(data_list[sidx], (int(np.ceil(time_steps / len(data_list[sidx]))), 1))
#   for cidx in range(cycles):
#     data[sidx, time_steps * cidx:time_steps * (cidx + 1), :] = data_list_curr[0:time_steps]

# # reshaped_data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords))
# # print("getting affective fearures...")
# # data = common.get_affective_features(reshaped_data)[:, :, :]

# print("Saving data...")
# np.save(npy_path, data)

data = np.load(npy_path)
# data = np.load("test.npy")
clips = data

print("Orig shape: ", clips.shape)

clips = np.swapaxes(clips, 1, 2)

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

animation_plot([angry * 2], filename="angry", interval=100.15, predicted="angry")
