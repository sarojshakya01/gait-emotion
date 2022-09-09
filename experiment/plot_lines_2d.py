import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe


def plot_and_animate(animations):

  for ai in range(len(animations)):
    anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
    joints = anim[:, :]
    joints = joints.reshape((len(joints), -1, 2))
    # joints[:, :, 0] *= 10
    # joints[:, :, 1] *= 5

    animations[ai] = joints

  scale = 1.0 * ((len(animations)) / 2)

  fig, ax = plt.subplots()

  lines = []
  acolors = list(sorted(colors.cnames.keys()))[::-1]
  parents = np.array([0, 1, 2, 3, 3, 5, 6, 3, 8, 9, 1, 11, 12, 1, 14, 15]) - 1

  for ai, anim in enumerate(animations):
    lines.append([plt.plot([0, 0], [0, 0], color=acolors[ai], lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(anim.shape[1])])

  # glider = set([(coords[0], coords[1]) for i, coords in enumerate(animations[0][1])])  #(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)])

  # print(glider)

  # mat, = ax.plot(1000, 1000, 'o')

  def animate(i):

    changed = []

    for ai in range(len(animations)):

      # x, y = zip(*glider)
      # mat.set_data(x, y)
      # return mat,

      # offset = 25 * (ai - ((len(animations)) / 2))
      offset = -20
      for j in range(len(parents)):

        # print(animations[ai][i * 2, j, 0], animations[ai][i * 2, parents[j], 0], -animations[ai][i * 2, j, 1])
        # exit()
        if parents[j] != -1:
          sdata = [animations[ai][i * 2, j, 0], animations[ai][i * 2, parents[j], 0]],\
                  [animations[ai][i * 2, j, 1], animations[ai][i * 2, parents[j], 1]]
          s3dprop = [animations[ai][i * 2, j, 1], animations[ai][i * 2, parents[j], 1]]
          lines[ai][j].set_data(sdata)

          # lines[ai][j].set_3d_properties(s3dprop)

        changed += lines
        print(changed)

      return changed

  plt.tight_layout()

  ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)

  # To save the animation, use e.g.
  #
  # ani.save("movie.mp4")
  #
  # or
  #
  # writer = animation.FFMpegWriter(
  #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
  # ani.save("movie.mp4", writer=writer)

  plt.show()