import numpy as np

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from utils.Quaternions import Quaternions


def animation_plot(animations, filename=None, ignore_root=True, interval=33.33):

  for ai in range(len(animations)):
    anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
    joints, root_x, root_z, root_r = anim[:, :], anim[:, 0], anim[:, 1], anim[:, 2]
    joints = joints.reshape((len(joints), -1, 3))
    joints[:, :, 0] *= 10
    joints[:, :, 1] *= 5
    joints[:, :, 2] *= 10
    rotation = Quaternions.id(1)
    translation = np.array([[0, 0, 0]])
    if not ignore_root:
      for i in range(len(joints)):
        joints[i, :, :] = rotation * joints[i]
        joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
        joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

    animations[ai] = joints

  scale = 1.0 * ((len(animations)) / 2)

  fig = plt.figure(figsize=(6, 8))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim3d(-scale * 50, scale * 50)
  ax.set_zlim3d(0, scale * 40)
  ax.set_ylim3d(-scale * 50, scale * 50)

  lines = []
  acolors = list(sorted(colors.cnames.keys()))[::-1]
  parents = np.array([0, 1, 2, 3, 3, 5, 6, 3, 8, 9, 1, 11, 12, 1, 14, 15]) - 1

  for ai, anim in enumerate(animations):
    lines.append([plt.plot([0, 0], [0, 0], [0, 0], color=acolors[ai], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(anim.shape[1])])

  def animate(i):
    changed = []
    for ai in range(len(animations)):

      offset = 25 * (ai - ((len(animations)) / 2))
      offset = -20
      for j in range(len(parents)):

        if parents[j] != -1:
          sdata = [animations[ai][i * 2, j, 0] + offset, animations[ai][i * 2, parents[j], 0] + offset],\
                  [-animations[ai][i * 2, j, 2], -animations[ai][i * 2, parents[j], 2]]
          s3dprop = [animations[ai][i * 2, j, 1], animations[ai][i * 2, parents[j], 1]]
          lines[ai][j].set_data(sdata)
          lines[ai][j].set_3d_properties(s3dprop)

        changed += lines

      return changed

  plt.tight_layout()

  ani = animation.FuncAnimation(fig, animate, np.arange(len(animations[0]) // 2), interval=interval)
  print(filename)
  if filename != None:
    writergif = animation.PillowWriter(fps=30)
    ani.save(filename + ".gif", writer=writergif)

  try:
    plt.show()
    plt.save()
  except AttributeError as e:
    pass
