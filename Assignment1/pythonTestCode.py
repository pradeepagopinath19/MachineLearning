import random
import RegressionTree
import numpy as np
import math
import random
import pandas as pd
import time



def main():
    dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    dataset = pd.read_csv(dataset_url, header=None, sep=',')
    dataset = np.array(dataset)
    newDataSet = dataset[dataset[: , (len(dataset[0])-2)].argsort()]
    print(newDataSet)


if __name__ == '__main__':
    main()

    # i = 0.1
    # while i < 1:
    #     classification_threshold = i
    #     accuracy = evaluate_prediction_accuracy(y_predict, Y_test, classification_threshold)
    #     print("The accuracy here is", accuracy)
    #     print("classification_threshold is", classification_threshold)
    #     i += 0.1
    #     spam_accuracy.append(accuracy)
    # exit()