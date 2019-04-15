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


def extract_full_dataset():
    training_features = pd.read_csv("spam_polluted_no_missing/train_feature.txt", header=None, sep='\s+').values
    training_label = pd.read_csv("spam_polluted_no_missing/train_label.txt", header=None, sep='\s+').values
    testing_features = pd.read_csv("spam_polluted_no_missing/test_feature.txt", header=None, sep='\s+').values
    testing_label = pd.read_csv("spam_polluted_no_missing/test_label.txt", header=None, sep='\s+').values

    training_dataset = np.column_stack((training_features, training_label))
    testing_dataset = np.column_stack((testing_features, testing_label))

    return training_dataset, testing_dataset


def update_labels(dataset):
    for row_number in range(len(dataset)):
        if dataset[row_number][-1] == 0:
            dataset[row_number][-1] = -1


def adaboost_algo(dataset, y_train, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(dataset)) / len(dataset)

    dec_classifiers = []

    for iter_number in range(max_iter):

        classifier = DecisionStump()
        min_weighted_error = math.inf

        # Best decision stump
        for j in range(len(dataset[0]) - 1):

            f_values = dataset[:, j]
            unique_feature = set(f_values)

            for threshold in unique_feature:
                # stump_prediction = make_prediction_from_stump(f_values, threshold)
                stump_prediction = np.ones((np.shape(y_train)))
                stump_prediction[f_values < threshold] = -1

                weighted_error = np.sum(w[y_train != stump_prediction])

                if weighted_error > 0.5:
                    p = -1
                    weighted_error = 1 - weighted_error
                else:
                    p = 1

                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error

                    classifier.threshold = threshold
                    classifier.feature = j
                    classifier.polarity = p
        classifier.alpha = 0.5 * math.log((1.0 - min_weighted_error) / (min_weighted_error + 1e-10))

        predictions = np.ones(y_train.shape)
        negative_idx = (
                classifier.polarity * dataset[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w
        # print(w.shape, y_train.shape, predictions.shape)
        # print(type(w), type(y_train), type(predictions))
        w *= np.exp(-classifier.alpha * y_train * predictions)

        w /= np.sum(w)

        dec_classifiers.append(classifier)

        # Printing and verification after each step

        prediction_y_train = predict(dec_classifiers, dataset[:, 0:57])
        prediction_y_test = predict(dec_classifiers, testing_x)

        training_accuracy = evaluate_prediction_accuracy(y_train, prediction_y_train)
        testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)

        auc_val = roc_auc_score(testing_y, prediction_y_test)

        print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
              "Weighted error", min_weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
              1 - testing_accuracy,
              "AUC", auc_val)

    return dec_classifiers


def predict(classifiers, X):
    y_pred = np.zeros((len(X), 1))

    for c in classifiers:
        non_spam_idx = (c.polarity * X[:, c.feature] < c.polarity * c.threshold)
        # print(non_spam_idx)

        predictions = np.ones((len(X), 1))
        predictions[non_spam_idx] = -1
        y_pred += c.alpha * predictions

    return np.sign(y_pred).flatten()


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues)


def main():
    number_iterations = 1
    training_dataset, testing_dataset = extract_full_dataset()

    print(training_dataset.shape, testing_dataset.shape)
    # shuffle
    training_dataset = shuffle(training_dataset)

    # {-1, +1}

    update_labels(training_dataset)
    update_labels(testing_dataset)

    # Converting to float

    # Training
    X = training_dataset[:, 0:len(training_dataset[0]) - 1]
    X = np.array(list(X[:, :]), dtype=np.float)

    y = training_dataset[:, -1]
    print(y)
    y = np.array(list(y), dtype=np.float)

    training_dataset = np.column_stack((X, y))

    # Testing
    X = testing_dataset[:, 0:len(testing_dataset[0]) - 1]
    X = np.array(list(X[:, :]), dtype=np.float)

    y = testing_dataset[:, -1]
    y = np.array(list(y), dtype=np.float)

    testing_dataset = np.column_stack((X, y))

    #     for i in range(len(training_dataset)):
    #         for j range(len(training_dataset[0])):
    #             print(training_dataset[i])
    training_x = training_dataset[:, 0:len(training_dataset[0]) - 1]
    training_y = training_dataset[:, -1]

    testing_x = testing_dataset[:, 0:len(testing_dataset[0]) - 1]
    testing_y = testing_dataset[:, -1]

    classifiers = adaboost_algo(training_dataset, training_y, testing_x, testing_y, number_iterations)

    prediction_y_train = predict(classifiers, training_x)
    prediction_y_test = predict(classifiers, testing_x)

    training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
    testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)

    print("Testing accuracy is:", testing_accuracy)
    print("Testing error rate is:", 1 - testing_accuracy)

    print("Training accuracy is:", training_accuracy)
    print("Training error rate is:", 1 - training_accuracy)


if __name__ == '__main__':
    main()
