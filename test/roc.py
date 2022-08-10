import torch
from torchmetrics import ROC
from cProfile import label
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

from torchmetrics.functional import f1_score

pred = torch.tensor([[194, 1, 1, 0], [2, 178, 58, 56], [4, 21, 141, 23], [0, 0, 0, 0]])

target = torch.tensor([0, 1, 3, 2])
roc_curve = ROC(num_classes=4)
fpr, tpr, thresholds = roc_curve(pred, target)

y_test = [
    1., 0., 3., 1., 0., 0., 0., 2., 0., 1., 0., 1., 0., 0., 3., 1., 0., 0., 2., 0., 2., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 3., 2., 1., 2., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 3., 1., 0., 0., 3., 1., 0., 2., 0., 0., 2., 2., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 3., 0., 1., 1., 0., 2., 2., 0., 1.,
    2., 0., 0., 0., 0., 2., 0., 1., 0., 1., 2., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 2., 0., 0., 0., 0., 2., 0., 2., 0., 0., 0., 2., 1., 1., 2., 0., 2., 0., 2., 1., 1., 0., 2., 1., 0.
]

y_score = [
    1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 2., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 2., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1.,
    1., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 2., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0.
]

print(f1_score(torch.tensor([int(x) for x in y_score]), torch.tensor([int(x) for x in y_test]), num_classes=4))

