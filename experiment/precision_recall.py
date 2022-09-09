import torch
from torchmetrics import PrecisionRecallCurve
from cProfile import label
import matplotlib.pyplot as plt

pred = torch.tensor([[993, 6.0, 11.0, 3.0], [47, 440, 169, 71], [8, 8, 74, 5], [0, 0, 0, 0]])

target = torch.tensor([0, 1, 3, 2])
pr_curve = PrecisionRecallCurve(num_classes=4)
precision, recall, thresholds = pr_curve(pred, target)

print(precision, recall)
# plt.plot(recall, precision)
# plt.legend(['Recall', 'Precision'])
# # naming the x axis
# plt.xlabel('epoch')
# # naming the y axis
# plt.ylabel('Loss')

# # function to show the plot
# plt.savefig("epoch_300.png")
# plt.show()
