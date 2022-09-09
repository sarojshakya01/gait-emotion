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

file_elmd_feautures = os.path.join(data_path, "features_ELMD.npy")
file_elmd_labels = os.path.join(data_path, "labels_ELMD.npy")

file_bhatta_features = os.path.join(data_path, "features_bhatta.npy")
file_bhatta_labels = os.path.join(data_path, "labels_bhatta.npy")

file_features_combined = os.path.join(data_path, "features_combined.npy")
file_labels_combined = os.path.join(data_path, "labels_combined.npy")

features = np.load(file_elmd_feautures)

print(features.shape)
# for i in list(np.load(file_bhatta_features)):
#   features.append(i)

# features = np.array(features)


labels = list(np.load(file_elmd_labels))

for i in np.load(file_bhatta_labels):
  labels.append(i)

labels = np.array(labels)

print(features.shape, labels.shape)

np.save(file_features_combined, features)
np.save(file_labels_combined, labels)
