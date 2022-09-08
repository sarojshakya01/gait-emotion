import h5py
import os
import numpy as np

from sklearn.model_selection import train_test_split
from utils import common

# torch
import torch


def load_data_npy(_path, _ftype, test_size=0.1):

  file_feature_npy = os.path.join(_path, 'features' + _ftype + '.npy')
  print("Loading data...", file_feature_npy)
  data = np.load(file_feature_npy)

  # file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
  # fl = h5py.File(file_label, 'r')
  # num_samples = len(fl.keys())
  # label = np.empty(num_samples)

  file_label_npy = os.path.join(_path, 'labels' + _ftype + '.npy')
  print("Loading data...", file_label_npy)
  label = np.load(file_label_npy)

  num_samples = label.shape[0]
  print("Total data: ", num_samples)
  new_label = []
  new_data = []

  # for sidx in range(num_samples):
  #   label[sidx] = fl[list(fl.keys())[sidx]][()]

  datacount = {'angry': 0, 'neutral': 0, 'happy': 0, 'sad': 0}
  count = 0
  for n in label:

    if n == 0:
      datacount['angry'] += 1
      if datacount['angry'] <= 5000:
        new_data.append(data[count])
        new_label.append(n)
    elif n == 1:
      datacount['neutral'] += 1
      if datacount['neutral'] <= 5000:
        new_data.append(data[count])
        new_label.append(n)
    elif n == 2:
      datacount['happy'] += 1
      if datacount['happy'] <= 5000:
        new_data.append(data[count])
        new_label.append(n)
    elif n == 3:
      datacount['sad'] += 1
      if datacount['sad'] <= 5000:
        new_data.append(data[count])
        new_label.append(n)
    count += 1

  print(datacount)

  datacount = {'angry': 0, 'neutral': 0, 'happy': 0, 'sad': 0}

  for n in new_label:
    if n == 0:
      datacount['angry'] += 1
    elif n == 1:
      datacount['neutral'] += 1
    elif n == 2:
      datacount['happy'] += 1
    elif n == 3:
      datacount['sad'] += 1

  print(datacount)

  new_data = np.array(new_data)
  new_label = np.array(new_label)

  data_train, data_test, labels_train, labels_test = train_test_split(new_data, new_label, test_size=test_size)
  print("Loading completed!!")
  return new_data, new_label, data_train, labels_train, data_test, labels_test


def load_data(_path, _ftype, coords, joints, cycles=3, test_size=0.1):

  file_feature = os.path.join(_path, 'features' + _ftype + '.h5')
  ff = h5py.File(file_feature, 'r')

  file_label = os.path.join(_path, 'labels' + _ftype + '.h5')
  fl = h5py.File(file_label, 'r')

  data_list = []
  num_samples = len(ff.keys())
  time_steps = 0
  labels = np.empty(num_samples)

  print("Loading data....")
  for sidx in range(num_samples):
    ff_group_key = list(ff.keys())[sidx]
    data_list.append(list(ff[ff_group_key]))  # Get the data
    time_steps_curr = len(ff[ff_group_key])
    # max steps
    if time_steps_curr > time_steps:
      time_steps = time_steps_curr
    labels[sidx] = fl[list(fl.keys())[sidx]][()]

  data = np.empty((num_samples, time_steps * cycles, joints * coords))

  for sidx in range(num_samples):
    data_list_curr = np.tile(data_list[sidx], (int(np.ceil(time_steps / len(data_list[sidx]))), 1))
    for cidx in range(cycles):
      data[sidx, time_steps * cidx:time_steps * (cidx + 1), :] = data_list_curr[0:time_steps]

  reshaped_data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords))
  data = common.get_affective_features(reshaped_data)[:, :, :48]
  data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
  return data, labels, data_train, labels_train, data_test, labels_test


def scale(_data):
  data_scaled = _data.astype('float32')
  data_max = np.max(data_scaled)
  data_min = np.min(data_scaled)
  data_scaled = (_data - data_min) / (data_max - data_min)
  return data_scaled, data_max, data_min


# descale generated data
def descale(data, data_max, data_min):
  data_descaled = data * (data_max - data_min) + data_min
  return data_descaled


def to_categorical(y, num_classes):
  """ 1-hot encodes a tensor """
  return np.eye(num_classes, dtype='uint8')[y]


class TrainTestLoader(torch.utils.data.Dataset):

  def __init__(self, data, label, joints, coords, num_classes):
    # data: N C T J
    self.data = np.reshape(data, (data.shape[0], data.shape[1], joints, coords, 1))
    self.data = np.moveaxis(self.data, [1, 2, 3], [2, 3, 1])

    # load label
    self.label = label

    self.N, self.C, self.T, self.J, self.M = self.data.shape

  def __len__(self):
    return len(self.label)

  def __getitem__(self, index):
    # get data
    data_numpy = np.array(self.data[index])
    label = self.label[index]

    # processing
    # if self.random_choose:
    #     data_numpy = tools.random_choose(data_numpy, self.window_size)
    # elif self.window_size > 0:
    #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
    # if self.random_move:
    #     data_numpy = tools.random_move(data_numpy)

    return data_numpy, label
