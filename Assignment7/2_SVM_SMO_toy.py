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

    return float(len(correct_predictions)) / len(actualValues) * 100


def extract_full_dataset():
    training_features = pd.read_csv("toy/train_x.csv", header=None, sep=',').values
    training_label = pd.read_csv("toy/train_y.csv", header=None, sep=',').values
    testing_features = pd.read_csv("toy/test_x.csv", header=None, sep=',').values
    testing_label = pd.read_csv("toy/test_y.csv", header=None, sep=',').values

    training_dataset = np.column_stack((training_features, training_label))
    testing_dataset = np.column_stack((testing_features, testing_label))

    return training_dataset, testing_dataset


def select_random_j_value(i, max_val):
    j = int(np.random.uniform(0, max_val))
    while j == i:
        j = int(np.random.uniform(0, max_val))

    return j


def calculate_eta(X, i, j):
    return 2.0 * X[i, :] * X[j, :].T - (X[i, :] * X[j, :].T) - X[j, :] * X[j, :].T
    # return (2 * np.dot(x_i, x_j)) - np.dot(x_i, x_i) - np.dot(x_j, x_j)


def calculate_alpha_j(alpha_j_old, y_j, e_i, e_j, eta, h, l):
    alpha_j = alpha_j_old - (((y_j) * (e_i - e_j)) / eta)

    if alpha_j > h:
        return h
    elif l > alpha_j:
        return l
    else:
        return alpha_j


def calculate_b1_b2(b, y_i, y_j, alpha_i, alpha_i_old, alpha_j, alpha_j_old, x_i, x_j, e_i, e_j):
    b1 = b - e_i - y_i * (alpha_i - alpha_i_old) * x_i * x_i.T - y_j * (alpha_j - alpha_j_old) * x_i * x_j.T
    b2 = b - e_j - y_i * (alpha_i - alpha_i_old) * x_i * x_j.T - y_j * (alpha_j - alpha_j_old) * x_j * x_j.T

    return b1, b2


def compute_final_b(alpha_i, alpha_j, c, b1, b2):
    if alpha_i > 0 and alpha_i < c:
        return b1
    elif alpha_j > 0 and alpha_j < c:
        return b2
    else:
        return (b1 + b2) / 2.0


def svm_smo(X, y):
    y = np.mat(y).transpose()
    X = np.mat(X)
    # Initialization
    c = 0.001
    tolerance = 0.01
    epsilon = 0.001
    number_of_iterations = 100
    alpha = np.mat(np.zeros((X.shape[0], 1)))
    b = 0
    # b = np.mat([[0]])
    # print(alpha, b)

    m, n = X.shape
    iter = 0
    while iter < number_of_iterations:
        alpha_changed = 0
        for i in range(m):
            # print("b values is", b)
            fxi = float(np.multiply(alpha, y).T * (X * X[i, :].T)) + b
            e_i = fxi - float(y[i])
            # e_i = np.multiply(y, alpha).T * X * X.T + b - y[i]
            # print(e_i)
            if ((y[i] * e_i < -tolerance) and (alpha[i] < c)) or ((y[i] * e_i > tolerance) and (alpha[i] > 0)):
                j = select_random_j_value(i, m)
                # print(i, j)

                fxj = float(np.multiply(alpha, y).T * (X * X[j, :].T)) + b
                e_j = fxj - float(y[j])

                # saving the old values - deep copy
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                if y[i] != y[j]:
                    l = max(0, alpha[j] - alpha[i])
                    h = min(c, c + alpha[j] - alpha[i])

                else:
                    l = max(0, alpha[j] + alpha[i] - c)
                    h = min(c, alpha[j] + alpha[i])

                if l == h:
                    continue

                eta = calculate_eta(X, i, j)
                if eta >= 0:
                    continue
                alpha[j] = calculate_alpha_j(alpha[j], y[j], e_i, e_j, eta, h, l)

                if abs(alpha[j] - alpha_j_old) < epsilon:
                    continue

                alpha[i] += y[j] * y[i] * (alpha_j_old - alpha[j])

                b1, b2 = calculate_b1_b2(b, y[i], y[j], alpha[i], alpha_i_old, alpha[j], alpha_j_old, X[i, :], X[j, :],
                                         e_i, e_j)
                b = compute_final_b(alpha[i], alpha[j], c, b1, b2)
                alpha_changed += 1
        if alpha_changed == 0:
            iter += 1
        else:
            iter = 0

        return alpha, b


def main():
    training_dataset, testing_dataset = extract_full_dataset()
    training_dataset = shuffle(training_dataset)

    X = training_dataset[:, 0:-1]
    y = training_dataset[:, -1]

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

    trainingSet_X = trainingSet[:, 0:-1]
    trainingSet_Y = trainingSet[:, -1]
    # print("Shape of X", trainingSet_X.shape)
    # print("Shape of Y", trainingSet_Y.shape)
    alpha, bias = svm_smo(trainingSet_X, trainingSet_Y)
    # print(alpha, bias)

    # Predicting values based on alpha and bias values
    testingSet_X = testingSet[:, 0:-1]
    testingSet_y = testingSet[:, -1]
    y = np.mat(trainingSet_Y).transpose()
    X = np.mat(trainingSet_X)
    X_test = np.mat(testingSet_X)

    predictions = []
    for i in range(len(X_test)):
        y_prediction = float(np.multiply(alpha, y).T * (X * X_test[i, :].T)) + bias
        # print(y_prediction)
        if y_prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    accuracy = evaluate_prediction_accuracy(predictions, testingSet_y)
    print("Accuracy is", accuracy)


if __name__ == '__main__':
    main()
