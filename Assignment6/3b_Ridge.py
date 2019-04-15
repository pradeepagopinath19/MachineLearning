import pandas as pd
import numpy as np
import numbers
import re
import random
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from random import randrange
from DecisionStump import DecisionStump
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def extract_full_dataset():
    training_features = pd.read_csv("spam_polluted_no_missing/train_feature.txt", header=None, sep='\s+').values
    training_label = pd.read_csv("spam_polluted_no_missing/train_label.txt", header=None, sep='\s+').values
    testing_features = pd.read_csv("spam_polluted_no_missing/test_feature.txt", header=None, sep='\s+').values
    testing_label = pd.read_csv("spam_polluted_no_missing/test_label.txt", header=None, sep='\s+').values

    training_dataset = np.column_stack((training_features, training_label))
    testing_dataset = np.column_stack((testing_features, testing_label))

    return training_dataset, testing_dataset


def main():
    training_dataset, testing_dataset = extract_full_dataset()

    print(training_dataset.shape, testing_dataset.shape)
    # shuffle
    training_dataset = shuffle(training_dataset)

    X_train = training_dataset[:, 0:-1]
    y_train = training_dataset[:, -1]

    X_test = testing_dataset[:, 0:-1]
    y_test = testing_dataset[:, -1]

    # scaler = preprocessing.StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    #
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(X_test)
    # X_test = scaler.transform(X_test)

    # Ridge regression
    clf = LogisticRegression(penalty='l2', solver='liblinear').fit(X_train, y_train)
    print("Training accuracy is", clf.score(X_train, y_train))
    print("Testing accuracy is", clf.score(X_test, y_test))

    # Lasso regression

    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    lasso.fit(X_train, y_train)
    print("Training accuracy is", lasso.score(X_train, y_train))
    print("Testing accuracy is", lasso.score(X_test, y_test))


if __name__ == '__main__':
    main()
