import numpy as np
import pandas as pd
import random
from DecisionStump import DecisionStump
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset


def kfold_split(k):
    kfold_list = random.sample(range(0, k), k)
    return kfold_list


def get_training_testing_split(dataset, split, index):
    k = len(split)
    len_of_k = len(dataset) // k
    starting_row = index * len_of_k
    ending_row = starting_row + len_of_k

    testing_data = dataset.iloc[starting_row:ending_row, :]
    training_data1 = dataset.iloc[0:starting_row, :]
    training_data2 = dataset.iloc[ending_row:len(dataset), :]
    training_data = training_data1.append(training_data2, sort=False)

    return training_data, testing_data


def make_prediction_from_stump(feature_values, threshold_val):
    prediction = []
    for i in range(len(feature_values)):
        if feature_values[i] < threshold_val:
            prediction.append(-1)
        else:
            prediction.append(1)

    return prediction


def adaboost_algo_random(dataset, y_train, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(dataset)) / len(dataset)

    dec_classifiers = []
    weighted_error = math.inf

    for iter_number in range(max_iter):
        classifier = DecisionStump()

        feature = randrange(0, len(dataset[0]) - 2)
        f_values = dataset[:, feature]
        unique_feature = set(f_values)
        unique_feature = list(unique_feature)
        random_index = randrange(len(unique_feature))
        threshold_val = unique_feature[random_index]
        stump_prediction = np.ones((np.shape(y_train)))
        stump_prediction[f_values < threshold_val] = -1
        weighted_error = np.sum(w[y_train != stump_prediction])

        if weighted_error > 0.5:
            p = -1
            weighted_error = 1 - weighted_error
        else:
            p = 1
        classifier.threshold = threshold_val
        classifier.feature = feature
        classifier.polarity = p
        classifier.alpha = 0.5 * math.log((1.0 - weighted_error) / (weighted_error + 1e-10))

        predictions = np.ones(y_train.shape)
        negative_idx = (
                classifier.polarity * dataset[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w

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
              "Weighted error", weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
              1 - testing_accuracy,
              "AUC", auc_val)

    return dec_classifiers


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
        #print(w.shape, y_train.shape, predictions.shape)
        print(type(w), type(y_train), type(predictions))
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


def main():
    dataset = extract_full_dataset()
    dataset = dataset.values
    X = dataset[:, 0:57]
    y = dataset[:, -1]

    spam_dataset = shuffle(dataset)

    dataset_k_split = kfold_split(2)
    number_iterations = 100
    random_number_iterations = 1750

    # for i in dataset_k_split:
    # trainingSet, testingSet = get_training_testing_split(spam_dataset, dataset_k_split, i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    trainingSet = np.column_stack((X_train, y_train))
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

    training_x = trainingSet[:, 0:57]
    training_y = trainingSet[:, -1]

    testing_x = testingSet[:, 0:57]
    testing_y = testingSet[:, -1]

    classifiers = adaboost_algo(trainingSet, training_y, testing_x, testing_y, number_iterations)

    # classifiers = adaboost_algo_random(trainingSet, training_y, testing_x, testing_y, random_number_iterations)

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
