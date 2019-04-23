import numpy as np
import pandas as pd
import random
from DecisionStump_features import DecisionStumpFeatures
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange
import matplotlib.pyplot as plt
import operator


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset.values


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues)


def predict(classifiers, X):
    y_pred = np.zeros((len(X), 1))

    for c in classifiers:
        non_spam_idx = (c.polarity * X[:, c.feature] < c.polarity * c.threshold)
        # print(non_spam_idx)

        predictions = np.ones((len(X), 1))
        predictions[non_spam_idx] = -1
        y_pred += c.alpha * predictions

    return np.sign(y_pred).flatten()


def adaboost_algo(dataset, y_train, testing_x, testing_y, max_iter):
    # For plotting
    auc_values = []
    testing_error_values = []
    training_error_values = []
    round_error_values = []

    # Initialize weights to 1/n initially
    w = np.ones(len(dataset)) / len(dataset)

    dec_classifiers = []

    for iter_number in range(max_iter):
        print("Decision stump", iter_number)
        classifier = DecisionStumpFeatures()
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
        classifier.predictions = np.sum(np.exp(-classifier.alpha * y_train * predictions))
        dec_classifiers.append(classifier)

        # # Printing and verification after each step
        #
        # prediction_y_train = predict(dec_classifiers, dataset[:, 0:57])
        # prediction_y_test = predict(dec_classifiers, testing_x)
        #
        # training_accuracy = evaluate_prediction_accuracy(y_train, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)
        #
        # auc_val = roc_auc_score(testing_y, prediction_y_test)
        #
        # print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
        #       "Weighted error", min_weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
        #       1 - testing_accuracy,
        #       "AUC", auc_val)

    return dec_classifiers


def fetch_top_fifteen(dict):
    sorted_x = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x[:15]


def fetch_best_features_margin(classifiers):
    feature_margin = {}
    for c in classifiers:
        if c.feature not in feature_margin:
            feature_margin[c.feature] = c.predictions
        else:
            feature_margin[c.feature] += c.predictions

    # sort and fetch top 15 features
    best_features = fetch_top_fifteen(feature_margin)
    print(len(best_features))

    return best_features


def fetch_best_features_alpha(classifiers):
    feature_alpha_dictionary = {}
    for c in classifiers:
        if c.feature not in feature_alpha_dictionary:
            feature_alpha_dictionary[c.feature] = c.alpha
        else:
            feature_alpha_dictionary[c.feature] += c.alpha

    # sort and fetch top 15 features
    best_features = fetch_top_fifteen(feature_alpha_dictionary)
    print(len(best_features))

    return best_features


def main():
    num_of_iterations = 100
    expected_output = [52, 51, 56, 15, 6, 22, 23, 4, 26, 24, 7, 54, 5, 19, 18]
    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    trainingSet = np.column_stack((X_train, y_train))
    print(trainingSet.shape)
    testingSet = np.column_stack((X_test, y_test))

    # {1,-1}

    training_y_col = len(trainingSet[0]) - 1

    for row_no_training in range(len(trainingSet)):
        if trainingSet[row_no_training][training_y_col] == 0:
            trainingSet[row_no_training][training_y_col] = -1

    testing_y_col = len(testingSet[0]) - 1

    for row_no_testing in range(len(testingSet)):
        if testingSet[row_no_testing][testing_y_col] == 0:
            testingSet[row_no_testing][testing_y_col] = -1

    training_x = trainingSet[:, 0:-1]
    training_y = trainingSet[:, -1]

    testing_x = testingSet[:, 0:-1]
    testing_y = testingSet[:, -1]

    classifiers = adaboost_algo(trainingSet, training_y, testing_x, testing_y, num_of_iterations)

    #best_features = fetch_best_features_alpha(classifiers)
    best_features = fetch_best_features_margin(classifiers)
    count = 0
    for feature in best_features:
        f = feature[0]
        print(f)
        if f in expected_output:
            count += 1
    print("Correct values", count)
    print("The best features are ", best_features)
    print("The accuracy is", (count / 15) * 100)


if __name__ == '__main__':
    main()
