
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os


current_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load("test.npy")
data = np.load(os.path.join(current_dir, "../data/features_bhatta.npy"))


data = data[0:1]
clips = data

total_frames = len(clips[0])
print("Orig shape: ", clips.shape)

clips = np.swapaxes(clips, 1, 2)

print("After swap: ", clips.shape)
seq1 = clips[0:1]

animations = [seq1]

for ai in range(len(animations)):
  anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
  joints = anim[:, :]
  joints = joints.reshape((len(joints), -1, 2))
  joints[:, :, 0] *= 1000
  joints[:, :, 1] *= 1000

  animations[ai] = joints

gliders = []
for i, frame in enumerate(animations[0]):
  glider = set([(coords[0], coords[1]) for i, coords in enumerate(frame)])
  gliders.append(glider)

fig, ax = plt.subplots()


def drawLines(frame):

  mat, = ax.plot(1000, 1000, 'o')
  # for i, coord in enumerate(frame):
  x_values = [frame[0, 0], frame[1, 0]]
  y_values = [-frame[0, 1], -frame[1, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[1, 0], frame[2, 0]]
  y_values = [-frame[1, 1], -frame[2, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[2, 0], frame[3, 0]]
  y_values = [-frame[2, 1], -frame[3, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[3, 0], frame[4, 0]]
  y_values = [-frame[3, 1], -frame[4, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[1, 0], frame[5, 0]]
  y_values = [-frame[1, 1], -frame[5, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[5, 0], frame[6, 0]]
  y_values = [-frame[5, 1], -frame[6, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[6, 0], frame[7, 0]]
  y_values = [-frame[6, 1], -frame[7, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[1, 0], frame[8, 0]]
  y_values = [-frame[1, 1], -frame[8, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[8, 0], frame[9, 0]]
  y_values = [-frame[8, 1], -frame[9, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[8, 0], frame[12, 0]]
  y_values = [-frame[8, 1], -frame[12, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[9, 0], frame[10, 0]]
  y_values = [-frame[9, 1], -frame[10, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[10, 0], frame[11, 0]]
  y_values = [-frame[10, 1], -frame[11, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[12, 0], frame[13, 0]]
  y_values = [-frame[12, 1], -frame[13, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[13, 0], frame[14, 0]]
  y_values = [-frame[13, 1], -frame[14, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[1, 0], frame[15, 0]]
  y_values = [-frame[1, 1], -frame[15, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")


def drawLines2(frame):

  mat, = ax.plot(1000, 1000, 'o')
  # for i, coord in enumerate(frame):
  x_values = [frame[0, 0], frame[1, 0]]
  y_values = [-frame[0, 1], -frame[1, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[1, 0], frame[2, 0]]
  y_values = [-frame[1, 1], -frame[2, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[2, 0], frame[3, 0]]
  y_values = [-frame[2, 1], -frame[3, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[2, 0], frame[4, 0]]
  y_values = [-frame[2, 1], -frame[4, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[4, 0], frame[5, 0]]
  y_values = [-frame[5, 1], -frame[5, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[5, 0], frame[6, 0]]
  y_values = [-frame[5, 1], -frame[6, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[2, 0], frame[7, 0]]
  y_values = [-frame[2, 1], -frame[7, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[7, 0], frame[8, 0]]
  y_values = [-frame[7, 1], -frame[8, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[8, 0], frame[9, 0]]
  y_values = [-frame[8, 1], -frame[9, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[0, 0], frame[10, 0]]
  y_values = [-frame[0, 1], -frame[10, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[10, 0], frame[11, 0]]
  y_values = [-frame[10, 1], -frame[11, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[11, 0], frame[12, 0]]
  y_values = [-frame[11, 1], -frame[12, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[0, 0], frame[13, 0]]
  y_values = [-frame[0, 1], -frame[13, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[13, 0], frame[14, 0]]
  y_values = [-frame[13, 1], -frame[14, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")

  x_values = [frame[14, 0], frame[15, 0]]
  y_values = [-frame[14, 1], -frame[15, 1]]

  if (0 not in x_values and 0 not in y_values):
    ax.plot(x_values, y_values, 'bo', linestyle="-")



mat, = ax.plot(1000, 1000, 'o')

# line, = plt.plot(0, 0,color='g',lw='0.5')

def animate(i):


  if i < total_frames:

    global glider
    x, y = zip(*gliders[i])
    print(x, y)
    frame = animations[0][i]
    ax.clear()
    ax.axis([-200, 800, -500, 400])
    drawLines2(frame)
    
    return mat,


ax.axis([-200, 800, -500, 400])
ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()