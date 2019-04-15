import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from scipy.stats import logistic
import math


def shift_scale_normalization(dataset):
    rows, cols = dataset.shape
    for col in range(cols - 2):
        dataset[:, col] -= abs(dataset[:, col]).min()

    for col in range(cols - 2):
        dataset[:, col] /= abs(dataset[:, col]).max()

    return pd.DataFrame.from_records(dataset)


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


def evaluate_prediction_accuracy(predictedValues, actualValues, classification_threshold):
    normalized_prediction = []
    for i in predictedValues:
        if i >= classification_threshold:
            normalized_prediction.append(1)
        else:
            normalized_prediction.append(0)
    correct_predictions = [i for i, j in zip(normalized_prediction, actualValues) if i == j]
    return len(correct_predictions) / len(actualValues) * 100, normalized_prediction


def evaluate_prediction(estimation, trueValues):
    errorValues = np.array(estimation) - np.array(trueValues)
    return (np.mean(np.square(errorValues)))


def main():
    trainingSet, testingSet = extract_full_dataset()

    # shuffle
    # trainingSet = shuffle(trainingSet)

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

    # Training error

    best_w = gradient_descent(x_train, y_train)

    # training error section
    x_b_training = np.c_[np.ones(x_train.shape[0]), x_train]
    train_y_predict = x_b_training.dot(best_w)
    train_y_sigmoid = sigmoid(train_y_predict)
    train_accu, train_normalized_prediction = evaluate_prediction_accuracy(train_y_sigmoid, y_train, 0.4)

    # testing error section

    # testing error section
    X_new_testing = np.c_[np.ones(x_test.shape[0]), x_test]
    test_y_predict = X_new_testing.dot(best_w)
    test_y_sigmoid = sigmoid(test_y_predict)
    print(test_y_predict)
    test_accu, test_normalized_prediction = evaluate_prediction_accuracy(test_y_sigmoid, y_test, 0.4)

    # print(y_test.shape, test_y_sigmoid.shape)
    y_test_list = []
    test_y_sigmoid_list = []

    for val in y_test:
        y_test_list.append(val)

    for val in test_y_sigmoid:
        test_y_sigmoid_list.append(val)

    print("Training error is", np.mean(train_accu))
    print("Testing accuracy is", np.mean(test_accu))


def gradient_descent(X, y):
    alpha = 0.01
    iterations = 1000
    m = X.shape[0]
    n = X.shape[1]

    X = np.c_[np.ones((len(X), 1)), X]
    limit = math.sqrt(1 / n)
    w = np.random.uniform(-limit, limit, size=(X.shape[1], 1))

    h_w = sigmoid(X.dot(w))

    for _ in range(iterations):
        w = w + ((alpha / m) * (X.T.dot(y - h_w)))
    return w


def sigmoid(z_list):
    z_list = np.array(z_list, dtype=np.float64)
    return (1 / (1 + np.exp(-z_list)))


if __name__ == '__main__':
    main()
