import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from random import shuffle, seed
from sklearn.model_selection import train_test_split
from numpy import array
from matplotlib import pylab
import math


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

    training_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt"
    testing_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt"
    training_data = pd.read_csv(training_url, header=None, sep='\s+')
    testing_data = pd.read_csv(testing_url, header=None, sep='\s+')

    #testing_data = shift_scale_normalization(testing_data.values)
    #training_data = shift_scale_normalization(training_data.values)

    return spam_dataset, training_data.values, testing_data.values


def calculate_objective(diff_error):
    return ((np.sum(np.square(diff_error))) * (1/ 2 * diff_error.shape[0]))


def gradientDescent_lr(x, y):
    w = np.random.normal(size=(x.shape[1], 1))
    learning_rate = 0.001
    #j_new =0
    for i in range(3000):
        # print(w.shape)
        h = x.dot(w)
        #print(h,y)
        # print(h.shape)
        # print(y.shape)
        error = np.subtract(h, y)
        #print(error)
        # j_old = j_new
        # j_new = calculate_objective(error)
        # if abs(j_old - j_new) < 0.01:
        #     break
        # print(error.shape)
        #print("Iteration %d, Error= %f" % (i, j_new))
        # print(w.shape)
        # print(error.shape)
        # print(x.shape)

        w = w - (learning_rate/ x.shape[0]) * x.T.dot(error)

    return w

def evaluate_prediction(estimation, trueValues):
    errorValues = np.array(estimation) - np.array(trueValues)
    return (np.mean(np.square(errorValues)))

def main():
    seed(1)
    spam_dataset, housing_training, housing_testing = extract_full_dataset()
    shuffle(spam_dataset.values)
    shuffle(housing_training)
    shuffle(housing_testing)
    # Housing section
    print(housing_testing.shape, housing_training.shape)

    y_test = housing_testing[:, -1]
    y_test = y_test[:, None]
    x_test = housing_testing[:, 0:len(housing_testing[0]) - 1]

    y_train = housing_training[:, -1]
    y_train = y_train[:, None]
    x_train = housing_training[:, 0:len(housing_training[0]) - 1]

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    train = scaler.transform(x_train)

    scaler.fit(x_test)
    test = scaler.transform(x_test)

    # Adding one to each instance
    x_b_training = np.c_[np.ones(train.shape[0]), train]
    best_w = gradientDescent_lr(x_b_training, y_train)
    #print(best_w)

    #training error section
    train_y_predict = x_b_training.dot(best_w)
    print("Train mse for housing is:", evaluate_prediction(train_y_predict, y_train))


    #testing error section
    X_new_b = np.c_[np.ones(test.shape[0]), test]
    test_y_predict = X_new_b.dot(best_w)
    #print(y_test)
    #print(test_y_predict)
    print("Test mse for housing is:", evaluate_prediction(test_y_predict, y_test))

if __name__ == '__main__':
    main()
