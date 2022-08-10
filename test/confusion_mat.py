from locale import normalize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import colors as mcolors
import numpy as np
import itertools

array = [[920, 10, 10, 10], [
    25,
    500,
    12,
    3,
], [
    14,
    26,
    700,
    12,
], [
    14,
    6,
    21,
    686,
]]


def plot_confusion_matrix2(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
  """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

  rcParams.update({'figure.autolayout': True})
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
    cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 7))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  if target_names is not None:
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if normalize:
      plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    else:
      plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

  plt.savefig("test.png")
  plt.show()


def confusion_matrix(confusion_matrix):
  df_cm = pd.DataFrame(confusion_matrix, index=[i for i in ['Angry', 'Neutral', 'Happy', 'Sad']], columns=[i for i in ['Angry', 'Neutral', 'Happy', 'Sad']])
  # plt.figure(figsize=(10,7))
  sn.set(font_scale=1.4)  # for label size
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

  plt.show()


def plot_confusion_matrix(confusion_matrix, title='CM', fontsize=50):
  mpl.style.use('seaborn')
  # rcParams['text.usetex'] = True
  rcParams['axes.titlepad'] = 20

  columns = ('Angry', 'Neutral', 'Happy', 'Sad')
  rows = columns
  fig, ax = plt.subplots()

  # Set colors
  colors = np.empty((4, 4))
  colors[0] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['goldenrod'], 1.0))
  colors[1] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['bisque'], 1.0))
  colors[2] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['paleturquoise'], 1.0))
  colors[3] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['limegreen'], 1.0))
  # colors[4] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightpink'], 1.0))
  # colors[5] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['hotpink'], 1.0))
  # colors[6] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['mistyrose'], 1.0))
  # colors[7] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lightsalmon'], 1.0))
  # colors[8] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['lavender'], 1.0))
  # colors[9] = np.array(mcolors.to_rgba(mcolors.CSS4_COLORS['cornflowerblue'], 1.0))

  n_rows = len(confusion_matrix)
  index = np.arange(len(columns)) + 0.3
  bar_width = 0.4

  # Initialize the vertical-offset for the stacked bar chart.
  y_offset = np.zeros(len(columns))

  # Plot bars and create text labels for the table
  cell_text = []
  for row in range(n_rows):
    # plt.bar(index, confusion_matrix[row], bar_width, bottom=y_offset,
    #                                                 color=colors[row])
    y_offset = y_offset + confusion_matrix[row]
    cell_text.append(['%d' % (x) for x in confusion_matrix[row]])

  # Add a table at the bottom of the axes
  the_table = plt.table(cellText=cell_text, rowLabels=rows, rowColours=colors, colLabels=columns, loc='bottom')

  the_table.set_fontsize(fontsize)
  the_table.scale(1, fontsize / 7)

  # Adjust layout to make room for the table:
  plt.subplots_adjust(left=0.2, bottom=0.1, top=0.99)

  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
  plt.ylabel(" predictions of each class", fontsize=fontsize)
  plt.xticks([])
  fig.savefig("C:\\Users\\saroj\\Documents\\ukg\\gait\\gait-emotion\\figures\\" + title + '.png', bbox_inches='tight')
  plt.show()


plot_confusion_matrix2(cm=np.array(array), normalize=False, target_names=['Angry', 'Neutral', 'Happy', 'Sad'], title="")
