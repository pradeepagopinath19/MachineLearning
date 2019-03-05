import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from random import shuffle, seed
from sklearn.model_selection import train_test_split
from numpy import array
from matplotlib import pylab
import math
import ConfusionMatrix

def shift_scale_normalization(dataset):
    rows, cols = dataset.shape
    for col in range(cols - 2):
        dataset[:, col] -= abs(dataset[:, col]).min()

    for col in range(cols - 2):
        dataset[:, col] /= abs(dataset[:, col]).max()

    return pd.DataFrame.from_records(dataset)


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    spam_dataset = shift_scale_normalization(spam_dataset.values)

    return spam_dataset


def evaluate_prediction_accuracy(predictedValues, actualValues, classification_threshold):
    normalized_prediction = []
    for i in predictedValues:
        if i >= classification_threshold:
            normalized_prediction.append(1)
        else:
            normalized_prediction.append(0)
    correct_predictions = [i for i, j in zip(normalized_prediction, actualValues) if i == j]
    return len(correct_predictions) / len(actualValues) * 100, normalized_prediction


def kfold_split(k):
    kfold_list = random.sample(range(0, k), k)
    return kfold_list


def get_training_testing_split(dataset, split, index):
    k = len(split)
    len_of_k = len(dataset) // k
    starting_row = index * len_of_k
    ending_row = starting_row + len_of_k
    # print(starting_row, ending_row)
    testing_data = dataset.iloc[starting_row:ending_row, :]
    training_data1 = dataset.iloc[0:starting_row, :]
    training_data2 = dataset.iloc[ending_row:len(dataset), :]
    training_data = training_data1.append(training_data2, sort=False)
    # print(testing_data)
    # print(dataset.shape, testing_data.shape, training_data.shape)
    return training_data, testing_data


def main():
    seed(1)
    spam_dataset = extract_full_dataset()
    shuffle(spam_dataset.values)
    # print(spam_dataset.shape)

    trainingSet, testingSet = train_test_split(spam_dataset, test_size=0.2)

    dataset_k_split = kfold_split(5)
    spam_accuracy = []
    train_accuracy = []

    #trainingSet, testingSet = get_training_testing_split(spam_dataset, dataset_k_split, i)
    trainingSet = trainingSet.values
    testingSet = testingSet.values

    # print(trainingSet.shape, testingSet.shape)
    y_test = testingSet[:, -1]
    y_test = y_test[:, None]
    x_test = testingSet[:, 0:len(testingSet[0]) - 1]

    y_train = trainingSet[:, -1]
    y_train = y_train[:, None]
    x_train = trainingSet[:, 0:len(trainingSet[0]) - 1]

    # print(x_train.shape, x_test.shape, y_test.shape, x_test.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    train = scaler.transform(x_train)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_test)
    test = scaler.transform(x_test)

    # Training error

    best_w = gradient_descent(train, y_train)

    # training error section
    x_b_training = np.c_[np.ones(train.shape[0]), train]
    train_y_predict = x_b_training.dot(best_w)
    train_y_sigmoid = sigmoid(train_y_predict)
    train_accu, train_normalized_prediction = evaluate_prediction_accuracy(train_y_sigmoid, y_train, 0.4)
    train_accuracy.append(train_accu)

    # testing error section

    # testing error section
    X_new_testing = np.c_[np.ones(test.shape[0]), test]
    test_y_predict = X_new_testing.dot(best_w)
    test_y_sigmoid = sigmoid(test_y_predict)
    # print(test_y_predict)
    test_accu, test_normalized_prediction = evaluate_prediction_accuracy(test_y_sigmoid, y_test, 0.4)
    spam_accuracy.append(test_accu)

    #print(y_test.shape, test_y_sigmoid.shape)
    y_test_list =[]
    test_y_sigmoid_list =[]

    for val in y_test:
        y_test_list.append(val)

    for val in test_y_sigmoid:
        test_y_sigmoid_list.append(val)

    ConfusionMatrix.confusion_matrix(y_test_list, test_normalized_prediction, test_y_sigmoid, True)
    print("Training error is", np.mean(train_accuracy))
    print("Testing accuracy is", np.mean(spam_accuracy))


def gradient_descent(X, y):
    alpha = 0.01
    iterations = 1500
    n = X.shape[0]

    X = np.c_[np.ones((len(X), 1)), X]
    w = np.random.normal(size=(X.shape[1], 1))

    h_w = sigmoid(X.dot(w))

    for _ in range(iterations):
        w = w + ((alpha / n) * (X.T.dot(y - h_w)))
    return w


def sigmoid(z_list):
    # print(z_list)
    z_list = np.array(z_list, dtype=np.float64)
    return (1.0 / (1 + np.exp(-z_list)))


if __name__ == '__main__':
    main()
