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
from sklearn import svm


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100


def extract_haar_features():
    #training_dataset = pd.read_csv("Haar_feature_training.csv", header=None, sep=',')
    training_dataset = pd.read_csv("Haar_feature_full_training.csv", header=None, sep=',')
    testing_dataset = pd.read_csv("Haar_feature_testing.csv", header=None, sep=',')
    return training_dataset.values, testing_dataset.values


def main():
    training_data, testing_data = extract_haar_features()
    training_data = shuffle(training_data)
    X_train = training_data[:, 0:-1]
    y_train = training_data[:, -1]
    X_test = testing_data[:, 0:-1]
    y_test = testing_data[:, -1]

    clf = svm.LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Testing accuracy is", calculate_accuracy(prediction, y_test))


if __name__ == '__main__':
    main()
