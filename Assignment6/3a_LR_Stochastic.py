import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from scipy.stats import logistic
import math


def extract_full_dataset():
    training_features = pd.read_csv("spam_polluted_no_missing/train_feature.txt", header=None, sep='\s+').values
    training_label = pd.read_csv("spam_polluted_no_missing/train_label.txt", header=None, sep='\s+').values
    testing_features = pd.read_csv("spam_polluted_no_missing/test_feature.txt", header=None, sep='\s+').values
    testing_label = pd.read_csv("spam_polluted_no_missing/test_label.txt", header=None, sep='\s+').values

    training_dataset = np.column_stack((training_features, training_label))
    testing_dataset = np.column_stack((testing_features, testing_label))

    # training_dataset = shift_scale_normalization(training_dataset).values
    # testing_dataset = shift_scale_normalization(testing_dataset).values

    return training_dataset, testing_dataset


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def predict(X, w):
    y_pred = w[0]
    for i in range(len(X)):
        y_pred += w[i + 1] * X[i]
    return sigmoid(y_pred)


def logistic_regression_stochastic(x_train, y_train, alpha, number_of_iterations):
    w = [0.0 for _ in range(len(x_train[0]) + 1)]
    for _ in range(number_of_iterations):
        for X, y in zip(x_train, y_train):
            y_pred = predict(X, w)
            error = y - y_pred
            w[0] = w[0] + alpha * error * y_pred * (1 - y_pred)
            for i in range(len(X)):
                w[i + 1] = w[i + 1] + alpha * error * y_pred * (1 - y_pred) * X[i]
    return w

def evaluate_prediction_accuracy(predictedValues, actualValues, classification_threshold):
    normalized_prediction = []
    for i in predictedValues:
        if i >= classification_threshold:
            normalized_prediction.append(1)
        else:
            normalized_prediction.append(0)
    correct_predictions = [i for i, j in zip(normalized_prediction, actualValues) if i == j]
    return len(correct_predictions) / len(actualValues) * 100, normalized_prediction

def main():
    trainingSet, testingSet = extract_full_dataset()

    # shuffle
    trainingSet = shuffle(trainingSet)

    # print(trainingSet.shape, testingSet.shape)
    y_test = testingSet[:, -1]
    y_test = y_test[:, None]
    x_test = testingSet[:, 0:len(testingSet[0]) - 1]

    y_train = trainingSet[:, -1]
    y_train = y_train[:, None]
    x_train = trainingSet[:, 0:len(trainingSet[0]) - 1]

    print(x_train.shape, x_test.shape, y_test.shape, x_test.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    alpha = 0.01
    number_of_iterations = 150

    best_weights = logistic_regression_stochastic(x_train, y_train, alpha, number_of_iterations)
    print(best_weights)

    # Testing

    y_prediction_model = []
    for X, y in zip(x_test, y_test):
        y_pred = predict(X, best_weights)
        #y_pred = round(float(y_pred))
        y_prediction_model.append(y_pred)


    # Checking accuracy
    accuracy = evaluate_prediction_accuracy(y_prediction_model, y_test, classification_threshold= 0.5)
    print("Testing accuracy is", accuracy)




if __name__ == '__main__':
    main()