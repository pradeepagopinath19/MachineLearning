import numpy as np
import pandas as pd
import random

import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange
import matplotlib.pyplot as plt
import operator


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues)


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset.values


def algorithm_prediction(alpha_i, b, trainingSet):
    print(alpha_i.shape, trainingSet.shape)
    return np.dot(alpha_i.T, trainingSet) + b


def calculate_w(alpha, X, y):
    return np.dot(X.T, np.multiply(alpha, y))


def calculate_b(X, y, w):
    b = y - np.dot(w.T, X.T)
    return np.mean(b)


def error_prediction(X, y, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int) - y


def select_random_j_value(i, max_val):
    j = np.random.randint(0, max_val, dtype=int)

    while j == i:
        j = np.random.randint(0, max_val, dtype=int)

    return j


def compute_l_h(alpha_i, alpha_j, y_i, y_j, c):
    if y_i != y_j:
        l = max(0, alpha_j - alpha_i)
        h = min(c, c + alpha_j - alpha_i)
    else:
        l = max(0, alpha_i + alpha_j - c)
        h = min(c, alpha_i + alpha_j)
    return l, h


def calculate_tow(x_i, x_j):
    return (2 * np.dot(x_i, x_j)) - np.dot(x_i, x_i) - np.dot(x_j, x_j)


def calculate_alpha_j(alpha_j_old, y_j, e_i, e_j, tou, h, l):
    alpha_j = alpha_j_old - (((y_j) * (e_i - e_j)) / tou)

    if alpha_j > h:
        return h
    elif alpha_j >= l and alpha_j <= h:
        return alpha_j
    else:
        return l


def calculate_alpha_i(alpha_i_old, y_i, y_j, alpha_j_old, alpha_j):
    return alpha_i_old + ((y_i * y_j) * (alpha_j_old - alpha_j))


def calculate_b1_b2(b, y_i, y_j, alpha_i, alpha_i_old, alpha_j, alpha_j_old, x_i, x_j, e_i, e_j):
    b1 = b - e_i - (y_i * (alpha_i - alpha_i_old) * (np.dot(x_i, x_j))) - (
            y_j * (alpha_j - alpha_j_old) * np.dot(x_i, x_j))
    b2 = b - e_j - (y_i * (alpha_i - alpha_i_old) * np.dot(x_i, x_j)) - (
                y_j * (alpha_j - alpha_j_old) * np.dot(x_j, x_i))

    return b1, b2


def compute_final_b(alpha_i, alpha_j, c, b1, b2):
    if alpha_i > 0 and alpha_i < c:
        return b1
    elif alpha_j > 0 and alpha_j < c:
        return b2
    else:
        return (b1 + b2) / 2


def svm_smo(trainingSet):
    # Initialization
    c = 1.0
    tolerance = 0.001
    number_of_iterations = 1

    alpha = np.zeros((len(trainingSet), 1))
    # print(alpha_i)
    b = 0
    X = trainingSet[:, 0: -1]
    y = trainingSet[:, -1].reshape(-1, 1)
    # print(X.shape, y.shape)

    n = len(trainingSet)
    for _ in range(number_of_iterations):
        for i in range(n):
            w = calculate_w(alpha, X, y)  # alpha_i or alpha??

            b = calculate_b(X, y, w)
            x_i, y_i = X[i, :], y[i]
            e_i = error_prediction(x_i, y_i, w, b)
            # print(e_i)

            if ((y_i * e_i) < -tolerance and alpha[i] < c) or ((y_i * e_i) > tolerance and alpha[i] > 0):
                j = select_random_j_value(i, n - 1)
                x_j, y_j = X[j, :], y[j]
                e_j = error_prediction(x_j, y_j, w, b)
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                l, h = compute_l_h(alpha_i_old, alpha_j_old, y_i, y_j, c)
                if l == h:
                    continue
                tou = calculate_tow(x_i, x_j)
                if tou >= 0:
                    continue
                alpha_j = calculate_alpha_j(alpha_j_old, y_j, e_i, e_j, tou, h, l)
                if abs(alpha_j - alpha_j_old) < 10e-5:
                    continue
                alpha_i = calculate_alpha_i(alpha_i_old, y_i, y_j, alpha_j_old, alpha_j)
                b1, b2 = calculate_b1_b2(b, y_i, y_j, alpha_i, alpha_i_old, alpha_j, alpha_j_old, x_i, x_j, e_i, e_j)
                b = compute_final_b(alpha_i, alpha_j, c, b1, b2)
    return alpha, b


def main():
    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    # {1,-1}

    training_y_col = len(trainingSet[0]) - 1

    for row_no_training in range(len(trainingSet)):
        if trainingSet[row_no_training][training_y_col] == 0:
            trainingSet[row_no_training][training_y_col] = -1

    testing_y_col = len(testingSet[0]) - 1

    for row_no_testing in range(len(testingSet)):
        if testingSet[row_no_testing][testing_y_col] == 0:
            testingSet[row_no_testing][testing_y_col] = -1

    # print(trainingSet.shape, testingSet.shape)

    alphas, bias = svm_smo(trainingSet)
    print(alphas, bias)

if __name__ == '__main__':
    main()
