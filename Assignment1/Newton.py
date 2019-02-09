from sklearn import preprocessing
import numpy as np
import math
import pandas as pd
import time
from matplotlib import pylab
from pylab import *
import random

def evaluate_prediction(estimation, trueValues):
    errorValues = np.array(estimation) - np.array(trueValues)
    return (np.sum(np.square(errorValues)) / len(errorValues))

def extract_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')
    #np.random.permutation(spam_dataset)
    return spam_dataset

def kfold_split(k):
    kfold_list = random.sample(range(0, k), k)
    return kfold_list

def get_training_testing_split(dataset, split, index):
    dataset= preprocessing.normalize(dataset)
    dataset = pd.DataFrame.from_records(dataset)
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

def sigmoid(z):
  return (1 / (1 + np.exp(-z)))
  # print(value)
  # if -value > np.log(np.finfo(type(value)).max):
  #     return 0.0
  # a = np.exp(-value)
  # return 1.0 / (1.0 + a)
  max_q = max(0.0, np.max(q))
  rebased_q = q - max_q
  return np.exp(rebased_q - np.logaddexp(-max_q, np.logaddexp.reduce(rebased_q)))




def calculate_weights(x, y , w):
    y = np.matrix(y).T
    for i in range(100):
        h = sigmoid(x.dot(w))
        deviation = h - y
        #print("Shape of h", h.shape)
        #print("shape of y", y.shape)
        sk = h * (1-h)
        #print(sk.shape)
        sk = np.reshape(sk,(len(sk),))
        s = np.diag(sk)
        #print(x.shape)
        #print(s.shape)
        hs = x.T.dot(s.dot(x))
        #print("SSS",error.shape)
        gk = x.T.dot(deviation)
        #print(hs.shape, gk.shape)
        #print(gk)
        w= w- 0.01 * np.array(pinv(hs).dot(gk))
    #print(w)
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

def shift_scale_normalization(dataset):
    rows, cols = dataset.shape
    for col in range(cols-2):
        dataset[:, col] -= abs(dataset[:, col]).min()

    for col in range(cols-2):
        dataset[:, col] /= abs(dataset[:, col]).max()

    return pd.DataFrame.from_records(dataset)

def main():
    dataset = extract_dataset()
    dataset_k_split = kfold_split(5)
    spam_mse = []
    spam_accuracy = []
    training_accuracy =[]
    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(dataset, dataset_k_split, i)
        # trainingSet =trainingSet.values
        # testingSet = testingSet.values
        # Extracting the labels
        y = [row[-1] for row in trainingSet]

        # Deleting the labels and just getting the feature values
        x = np.delete(trainingSet, trainingSet.shape[1] - 1, axis=1)

        # Adding bias
        trainingSet = np.c_[np.ones(x.shape[0]), x]



        #print(y, trainingSet,x)

        w = np.random.normal(size=(trainingSet.shape[1], 1))

        updated_w = calculate_weights(trainingSet, y, w)
        #print(updated_w.shape)

        y_test = [row[-1] for row in testingSet]
        testingSet = np.delete(testingSet, -1, axis=1)
        X_new_b = np.c_[np.ones(testingSet.shape[0]), testingSet]

        y_predict = X_new_b.dot(updated_w)

        y_predict = sigmoid(y_predict)
        # accuracy_mse = evaluate_prediction(y_predict, y_test)
        # spam_mse.append(accuracy_mse)
        print(y_predict)
        print(y_test)
        # MSE calculation


        # accuracy
        accuracy = evaluate_prediction_accuracy(y_predict, y_test, classification_threshold= 0.38)
        spam_accuracy.append(accuracy)

        y_predict = trainingSet.dot(updated_w)

        y_predict = sigmoid(y_predict)

        train_accuracy =evaluate_prediction_accuracy(y_predict, y, classification_threshold= 0.38)
        training_accuracy.append(train_accuracy)
        #print(accuracy)
    print("Test accuracy for runs",spam_accuracy)
    print("Training accuracy for runs", training_accuracy)
    # print("MSE average is", np.mean(accuracy_mse))
    print("Test accuracy average is", np.mean(spam_accuracy))
    print("Training error average is", np.mean(training_accuracy))



if __name__ == "__main__":
    main()