import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from random import shuffle, seed
from numpy import array
from matplotlib import pylab
import math


def shift_scale_normalization(dataset):
    rows, cols = dataset.shape
    for col in range(cols - 2):
        dataset[:, col] -= abs(dataset[:, col]).min()

    for col in range(cols - 2):
        dataset[:, col] /= abs(dataset[:, col]).max()

    return dataset


def extract_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')
    # np.random.shuffle(spam_dataset.values)
    #df_shuffled = spam_dataset.reindex(np.random.permutation(spam_dataset.index))
    return spam_dataset.values


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
    return training_data.values, testing_data.values


def sigmoid(z_list):
    # print(z_list)
    z_list = np.array(z_list, dtype=np.float64)
    return (1 / (1 + np.exp(-z_list)))

def calculate_weights(x, y):
    x = np.c_[np.ones(x.shape[0]), x]
    # print(x_b)
    # print(x_b.shape)
    w = np.random.normal(size=(x.shape[1], 1))
    len_of_x = len(x)
    # y = np.matrix(y).T
    for i in range(100):
        h = sigmoid(x.dot(w))
        deviation = h - y
        # print("Shape of h", h.shape)
        # print("shape of y", y.shape)
        sk = h * (1 - h)
        # print(sk.shape)
        sk = np.reshape(sk, (len(sk),))
        s = np.diag(sk)
        # print(x.shape)
        # print(s.shape)
        hs = x.T.dot(s.dot(x))
        # print("SSS",error.shape)
        gk = x.T.dot(deviation)
        # print(hs.shape, gk.shape)
        # print(gk)
        w = w - 0.1 *np.array(np.linalg.pinv(hs).dot(gk))
    # print(w)
    return w


def evaluate_prediction_accuracy(predictedValues, actualValues, classification_threshold):
    normalized_prediction = []
    for i in predictedValues:
        if i >= classification_threshold:
            normalized_prediction.append(1)
        else:
            normalized_prediction.append(0)
    correct_predictions = [i for i, j in zip(normalized_prediction, actualValues) if i == j]
    return len(correct_predictions) / len(actualValues) * 100


def evaluate_prediction_accuracy_round(prediction, actual):
    correct_predictions = [i for i, j in zip(prediction, actual) if i == j]
    return len(correct_predictions) / len(prediction) * 100


def main():
    #seed(3)
    scaler = preprocessing.StandardScaler()
    dataset = extract_dataset()
    shuffle(dataset)
    # print(dataset.shape)

    y = dataset[:, -1]
    x = dataset[:, 0:dataset.shape[1] - 1]
    # print(x.shape, y.shape)

    # normalized_ds = shift_scale_normalization(dataset)
    scaler.fit(x)
    normalized_x = scaler.transform(x)

    # print(normalized_x)

    y = y[:, None]
    # print(y.shape)

    spam_dataset = np.append(normalized_x, y, 1)
    # print("checking",spam_dataset.shape)

    dataframe = pd.DataFrame.from_records(spam_dataset)
    dataset_k_split = kfold_split(2)
    spam_accuracy = []

    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(dataframe, dataset_k_split, i)
        # print(trainingSet.shape)
        # print(testingSet.shape)
        trainX = trainingSet[:, 0:trainingSet.shape[1] - 1]
        trainY = trainingSet[:, -1]
        trainY = trainY[:, None]

        testX = testingSet[:, 0:testingSet.shape[1] - 1]
        testY = testingSet[:, -1]
        testY = testY[:, None]

        # print(trainX.shape, trainY.shape)
        # print(testX.shape, testY.shape)

        updated_w = calculate_weights(trainX, trainY)
        # print(updated_w.shape)

        testX = np.c_[np.ones(testX.shape[0]), testX]
        y_predict = testX.dot(updated_w)
        y_sigmoid = np.round(sigmoid(y_predict))

        #accuracy = evaluate_prediction_accuracy_round(y_sigmoid, testY)

        accuracy = evaluate_prediction_accuracy(y_predict, testY, classification_threshold=0.38)
        spam_accuracy.append(accuracy)

    print("Accuracy for each trial", spam_accuracy)
    print("Mean accuracy", np.mean(spam_accuracy))


if __name__ == "__main__":
    main()
