import pandas as pd
import numpy as np
import numbers
import re
import random
from DecisionStump import DecisionStump
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from random import randrange
from Adaboost import adaboost_algo, adaboost_algo_random, predict, evaluate_prediction_accuracy


def extract_full_dataset():
    dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    dataset = pd.read_csv(dataset_url, header=None, sep=',')
    return dataset


def main():
    number_iterations = 100
    dataset = extract_full_dataset()

    # Data pre processing
    # Replace ? with np.nan values in X of the dataset
    dataset.replace("?", np.nan, inplace=True)

    # Replace nan with most frequent data in that feature
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dataset = imp_mean.fit_transform(dataset)

    # Label encoding to handle non numeric values
    lb_encoding = LabelEncoder()

    for col in range(len(dataset[0]) - 1):
        value = dataset[0][col]
        # Numeric type values are not encoded
        if type(value) == int or type(value) == float or '.' in value:
            continue
        if not value.isnumeric():
            dataset[:, col] = lb_encoding.fit_transform(dataset[:, col])

    # Shuffling
    dataset = shuffle(dataset)

    X = dataset[:, 0:len(dataset[0]) - 1]

    y = dataset[:, -1]

    for i in range(len(y)):
        if y[i] == '+':
            y[i] = +1
        else:
            y[i] = -1

    dataset = np.column_stack((X,y))

    for col in range(len(dataset[0])):
        dataset[:, col] = dataset[:, col].astype('float64')

    X = dataset[:, 0:len(dataset[0]) - 1]

    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    training_y_col = len(trainingSet[0]) - 1

    for row_no_training in range(len(trainingSet)):
        if trainingSet[row_no_training][training_y_col] == '+':
            trainingSet[row_no_training][training_y_col] = +1
        else:
            trainingSet[row_no_training][training_y_col] = -1

    testing_y_col = len(testingSet[0]) - 1

    for row_no_testing in range(len(testingSet)):
        if testingSet[row_no_testing][testing_y_col] == '+':
            testingSet[row_no_testing][testing_y_col] = +1
        else:
            testingSet[row_no_testing][testing_y_col] = -1

    training_x = trainingSet[:, 0:len(trainingSet[0]) - 1]
    training_y = trainingSet[:, -1]

    testing_x = testingSet[:, 0:len(trainingSet[0]) - 1]
    testing_y = testingSet[:, -1]

    classifiers = adaboost_algo(trainingSet, training_y, testing_x, testing_y, number_iterations)

    prediction_y_train = predict(classifiers, training_x)
    prediction_y_test = predict(classifiers, testing_x)

    training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
    testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)

    print("Testing accuracy is:", testing_accuracy)
    print("Testing error rate is:", 1 - testing_accuracy)

    print("Training accuracy is:", training_accuracy)
    print("Training error rate is:", 1 - training_accuracy)


if __name__ == '__main__':
    main()
