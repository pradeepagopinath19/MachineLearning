import pandas as pd
import numpy as np
import numbers
import re
import random
from DecisionStump import DecisionStump
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from random import randrange
from Adaboost import adaboost_algo, adaboost_algo_random, predict, evaluate_prediction_accuracy
import collections


def adaboost_algo(training_x, training_y, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(training_x)) / len(training_x)

    dec_classifiers = []

    for iter_number in range(max_iter):

        classifier = DecisionStump()
        min_weighted_error = math.inf

        # Best decision stump
        for j in range(len(training_x[0])):

            f_values = training_x[:, j]
            unique_feature = set(f_values)

            for threshold in unique_feature:
                stump_prediction = np.ones((np.shape(training_y)))
                stump_prediction[f_values < threshold] = -1

                weighted_error = np.sum(w[training_y != stump_prediction])

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

        predictions = np.ones(training_y.shape)
        negative_idx = (
                classifier.polarity * training_x[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w
        # print(w.shape, y_train.shape, predictions.shape)
        # print(type(w), type(y_train), type(predictions))
        w *= np.exp(-classifier.alpha * training_y * predictions)

        w /= np.sum(w)

        dec_classifiers.append(classifier)

        # Printing and verification after each step

        # prediction_y_train = predict(dec_classifiers, training_x)
        # prediction_y_test = predict(dec_classifiers, testing_x)
        #
        # training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)
        #
        # auc_val = roc_auc_score(testing_y, prediction_y_test)
        #
        # print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
        #       "Weighted error", min_weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
        #       1 - testing_accuracy,
        #       "AUC", auc_val)

    return dec_classifiers


def populate_data(data_collection):
    no_of_columns = 1755
    dataset = np.zeros((len(data_collection), no_of_columns))

    for row_index, data_point in enumerate(data_collection):
        for element in data_point:
            ele = element.split()
            dataset[row_index][-1] = float(ele[0])
            for index in range(1, len(ele), 1):
                feature_id, value = ele[index].split(":")
                dataset[row_index][int(feature_id)] = float(value)
    return dataset


def extract_full_dataset():
    news_training = pd.read_csv("feature_matrix_training.txt", header=None).values
    news_testing = pd.read_csv("feature_matrix_testing.txt", header=None).values

    training_data = populate_data(news_training)
    testing_data = populate_data(news_testing)

    return training_data, testing_data


def fetch_label_code():
    label_code = pd.read_csv("ECOC_Label_code.txt", header=None, sep=',').values

    return label_code


def minimum_dist(prediction, label_code):
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    for label in label_code:
        min_val = 21
        minimum_label_val = []
        diff = sum(1 for a, b in zip(label, prediction) if a != b)

        if diff < min_val:
            min_val = diff
            minimum_label_val = label

        if diff == 0:
            break

    for i, value in enumerate(label_code):
        if compare(value, minimum_label_val):
            return i


def fetch_minimum_distance(test_prediction, label_code):
    final_prediction = []

    for prediction in test_prediction:
        best_label = minimum_dist(prediction, label_code)
        final_prediction.append(best_label)
        # print(best_label)
    return final_prediction


def main():
    # Extracting dataset and label encoding
    number_iterations = 100

    training_data, testing_data = extract_full_dataset()
    training_data = shuffle(training_data)
    label_code = fetch_label_code()

    original_train_y = np.copy(training_data[:, -1])
    original_test_y = np.copy(testing_data[:, -1])

    train_label = []
    for val in training_data[:, -1]:
        train_label.append(label_code[int(val)])
    training_data = np.delete(training_data, -1, 1)
    training_data = np.column_stack((training_data, train_label))

    # print(training_data.shape)

    test_label = []
    for val in testing_data[:, -1]:
        test_label.append(label_code[int(val)])
    testing_data = np.delete(testing_data, -1, 1)
    testing_data = np.column_stack((testing_data, test_label))

    # print(testing_data.shape)

    # {1, -1}

    for row in range(len(training_data)):
        for col in range(-1, -21, -1):
            if training_data[row][col] == 0.0:
                training_data[row][col] = - 1.0

    for row in range(len(testing_data)):
        for col in range(-1, -21, -1):
            if testing_data[row][col] == 0.0:
                testing_data[row][col] = - 1.0
    print(testing_data.shape)

    training_x = np.copy(training_data[:, 0:-20])
    testing_x = np.copy(testing_data[:, 0:-20])

    train_prediction_array = []
    test_prediction_array = []

    for i in range(20, 0, -1):
        training_y = np.copy(training_data[:, -i])
        testing_y = np.copy(testing_data[:, -i])

        classifiers = adaboost_algo(training_x, training_y, testing_x, testing_y, number_iterations)
        #prediction_y_train = predict(classifiers, training_x)
        prediction_y_test = predict(classifiers, testing_x)

        #train_prediction_array.append(prediction_y_train)
        test_prediction_array.append(prediction_y_test)

        #training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)
        #
        # print("Testing accuracy for ", -i, "is:", testing_accuracy)
        # print("Testing error rate for ", -i, "is:", 1 - testing_accuracy)
        print("Round done:", i)

        # print("Training accuracy for ", -i, "is:", training_accuracy)
        # print("Training error rate for ", -i, "is:", 1 - training_accuracy)

    # Correct format - Transpose - Check if transpose is actually needed?
    test_prediction = np.array(test_prediction_array).T
    train_prediction_array = np.array(train_prediction_array).T

    print(test_prediction.shape)

    for i in range(len(test_prediction)):
        for j in range(len(test_prediction[0])):
            if test_prediction[i][j] == -1:
                test_prediction[i][j] = 0

    print("Predictions are")
    print(test_prediction)
    y_label_prediction = fetch_minimum_distance(test_prediction, label_code)

    # calculating test accuracy
    print(y_label_prediction)
    print(original_test_y)
    print(len(y_label_prediction))
    print(len(original_test_y))

    testing_accuracy = evaluate_prediction_accuracy(y_label_prediction, original_test_y)

    print("Testing accuracy is", testing_accuracy)


if __name__ == '__main__':
    main()
