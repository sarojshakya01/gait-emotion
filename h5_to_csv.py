from cProfile import label
import csv
from email import header
import h5py
import os
import numpy as np

root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, "data")
type = '4DCVAEGCN'  # '_ELMD'
feature_file_name = 'features' + type + '.h5'
label_file_name = 'labels' + type + '.h5'

file_path = os.path.join(data_path, feature_file_name)
label_path = os.path.join(data_path, label_file_name)

f = h5py.File(file_path, 'r')
fl = h5py.File(label_path, 'r')


emotions = ['Angry', 'Neutral', 'Happy', 'Sad']
header_row = ["SN",  "Emotion", "Cycles", "Root", "Spine", "Neck", "Head",
              "Left Shoulder", "Left Elbow", "Left Hand", "Right Shoulder", "Right Elbow", "Right Hand", "Left Hip", "Left Knee", "Left foot", "Right Hip", "Right Knee", "Right foot"]

with open('labeled_features.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header_row)
    for nth_feauture in range(len(f.keys())):
        a_group_key = list(fl.keys())[nth_feauture]
        a_group_key_f = list(f.keys())[nth_feauture]
        label_num = fl[a_group_key][()]
        if label_num > 3:
            continue
        print(emotions[label_num])
        feature = f[a_group_key_f][()]
        for nth_cycle in range(feature.shape[0]):
            a_cycle_feature = feature[nth_cycle]

            root = a_cycle_feature[0:3]
            spine = a_cycle_feature[3:6]
            neck = a_cycle_feature[6:9]
            head = a_cycle_feature[9:12]
            rshoulder = a_cycle_feature[12:15]
            relbow = a_cycle_feature[15:18]
            rhand = a_cycle_feature[18:21]
            lshoulder = a_cycle_feature[21:24]
            lelbow = a_cycle_feature[24:27]
            lhand = a_cycle_feature[27:30]
            rhip = a_cycle_feature[30:33]
            rknee = a_cycle_feature[33:36]
            rfoot = a_cycle_feature[36:39]
            lhip = a_cycle_feature[39:42]
            lknee = a_cycle_feature[42:45]
            lfoot = a_cycle_feature[45:48]

            data = [nth_feauture+1, emotions[label_num], nth_cycle+1,
                    "({:f},{:f},{:f})".format(root[0], root[1], root[2]),
                    "({:f},{:f},{:f})".format(spine[0], spine[1], spine[2]),
                    "({:f},{:f},{:f})".format(neck[0], neck[1], neck[2]),
                    "({:f},{:f},{:f})".format(head[0], head[1], head[2]),
                    "({:f},{:f},{:f})".format(
                        rshoulder[0], rshoulder[1], rshoulder[2]),
                    "({:f},{:f},{:f})".format(relbow[0], relbow[1], relbow[2]),
                    "({:f},{:f},{:f})".format(rhand[0], rhand[1], rhand[2]),
                    "({:f},{:f},{:f})".format(
                        lshoulder[0], lshoulder[1], lshoulder[2]),
                    "({:f},{:f},{:f})".format(lelbow[0], lelbow[1], lelbow[2]),
                    "({:f},{:f},{:f})".format(lhand[0], lhand[1], lhand[2]),
                    "({:f},{:f},{:f})".format(rhip[0], rhip[1], rhip[2]),
                    "({:f},{:f},{:f})".format(rknee[0], rknee[1], rknee[2]),
                    "({:f},{:f},{:f})".format(rfoot[0], rfoot[1], rfoot[2]),
                    "({:f},{:f},{:f})".format(lhip[0], lhip[1], lhip[2]),
                    "({:f},{:f},{:f})".format(lknee[0], lknee[1], lknee[2]),
                    "({:f},{:f},{:f})".format(lfoot[0], lfoot[1], lfoot[2])]

            writer.writerow(data)
