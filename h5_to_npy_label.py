from itertools import cycle
import numpy as np

from utils import plot_lines

import os
import h5py
from utils import common

root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, "data")
type = '_bhatta'  # '' or '4DCVAEGCN' or '_ELMD'

npy_path = os.path.join(data_path, "labels" + type + ".npy")

file_label = os.path.join(data_path, 'labels' + type + '.h5')
fl = h5py.File(file_label, 'r')
num_samples = len(fl.keys())
label = np.empty(num_samples)

for sidx in range(num_samples):
  label[sidx] = fl[list(fl.keys())[sidx]][()]

print("Saving data...")
np.save(npy_path, label)
print("Saved")