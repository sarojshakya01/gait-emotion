import enum
import json, os
from ntpath import join
import numpy as np

openpose_output = "C:/users/saroj/Downloads/openpose_gpu/output/keypoints/"
video_name = "man_side/"
json_path = openpose_output + video_name
number_of_people = 1  # len(json.load(open(json_path + os.listdir(json_path)[0]))['people'])
features = [[] for i in range(number_of_people)]
time_steps = 0
for f, filename in enumerate(os.listdir(json_path)):
  # clamp the ends
  if f <= 50:
    continue
  elif f > 290:
    break
  time_steps += 1
  print("Processing", f, "file:", filename)
  file = open(json_path + filename)
  data = json.load(file)

  exclude_joints = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

  for n in range(1):
    person = data['people'][0]
    # for n, person in enumerate(data['people']):
    selected_joints = []
    all_joints = np.array(person["pose_keypoints_2d"])
    all_joints = np.reshape(all_joints, (-1, 3))
    for i, joint in enumerate(all_joints):
      if i not in exclude_joints:
        selected_joints.append(list(joint[0:2]))

    spine_joint = list((np.array(selected_joints[1]) + np.array(selected_joints[8])) / 2)
    selected_joints.insert(15, spine_joint)
    # for i, joint in enumerate(selected_joints):
    #   selected_joints[i] = list(np.array(selected_joints[i]) / (np.full((1, 2), 100)))

    features[0].append(selected_joints)

  file.close()

features = np.array(features)
features = np.reshape(features, (number_of_people, time_steps, -1))
print(features.shape)
print(features[0, 0, :])
np.save("test.npy", features)
