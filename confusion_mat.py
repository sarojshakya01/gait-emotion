import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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

df_cm = pd.DataFrame(array, index=[i for i in ['Angry', 'Neutral', 'Happy', 'Sad']], columns=[i for i in ['Angry', 'Neutral', 'Happy', 'Sad']])
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

plt.show()