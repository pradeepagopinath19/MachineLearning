import pandas as pd
import ROC
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

def confusion_matrix(actual, predicted, raw_predicted, Roc):
    #print("Checking",predicted)
    y_actual = pd.Series(actual, name='Actual')
    y_actual = pd.Series.astype(y_actual, dtype='int64', copy=True)
    y_predicted = pd.Series(predicted, name='Predicted')
    #print(y_actual, y_predicted)
    df_confusion = pd.crosstab(y_actual, y_predicted)
    print(df_confusion)

    if Roc:
        ROC.plot_roc(raw_predicted, actual)