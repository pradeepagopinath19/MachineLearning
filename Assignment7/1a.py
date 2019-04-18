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


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset.values


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100


def main():
    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = svm.SVC(gamma='scale', kernel='rbf')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("RBF kernel Accuracy is", calculate_accuracy(prediction, y_test))

    clf = svm.SVC(gamma='scale', kernel='poly')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Polynomial kernel Accuracy is", calculate_accuracy(prediction, y_test))

    clf = svm.SVC(gamma='scale', kernel='sigmoid')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Sigmoid kernel Accuracy is", calculate_accuracy(prediction, y_test))

    clf = svm.SVC(gamma='scale', kernel='linear')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print("Linear kernel Accuracy is", calculate_accuracy(prediction, y_test))


if __name__ == '__main__':
    main()
