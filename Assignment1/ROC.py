
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def plot_roc(score, y):

    threshold = np.linspace(min(score), max(score), 40)
    #print(len(threshold))
    FP = 0
    TP = 0
    #FP + TN
    N = np.sum(y)

    #TP + FN
    P = len(y) - N

    P = 1 if P==0 else P
    N = 1 if N==0 else N
    #print(N,P)
    x_axis, y_axis = [], []

    for (i, T) in enumerate(threshold):
        for i in range(len(score)):
            if (score[i] > T):
                if (y[i] == 1):
                    TP = TP + 1
                if (y[i] == 0):
                    FP = FP + 1

        x_axis.append(FP / float(N))
        y_axis.append(TP / float(P))
        FP = 0
        TP = 0

    #print(x_axis, y_axis)
    plt.plot(x_axis, y_axis)
    plt.show()
    auc1 = metrics.auc(x_axis, y_axis)
    print("AUC score", auc1)
