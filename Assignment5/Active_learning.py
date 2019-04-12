import numpy as np
import pandas as pd
import random
from DecisionStump import DecisionStump
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange
import matplotlib.pyplot as plt


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


def predict_and_fetch_threshold_datapoints(classifiers, X):
    y_pred = np.zeros((len(X), 1))
    for c in classifiers:
        non_spam_idx = (c.polarity * X[:, c.feature] < c.polarity * c.threshold)
        predictions = np.ones((len(X), 1))
        predictions[non_spam_idx] = -1
        y_pred += c.alpha * predictions


    len_dataset_to_be_added = int(0.02 * len(y_pred))
    dataset_indices_to_be_added = []

    for _ in range(len_dataset_to_be_added):
        min_value = min(y_pred, key=abs)
        try:
            min_index = y_pred.tolist().index(min_value)
        except:
            min_index = y_pred.tolist().index(-min_value)
        dataset_indices_to_be_added.append(min_index)
        y_pred[min_index] = math.inf

    return dataset_indices_to_be_added


def main():
    number_iterations = 25

    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    # print(dataset.shape)

    # {1,-1}

    training_y_col = len(dataset[0]) - 1

    for row_no_training in range(len(dataset)):
        if dataset[row_no_training][training_y_col] == 0:
            dataset[row_no_training][training_y_col] = -1

    X = dataset[:, 0:57]
    y = dataset[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    # initial 5 percent dataset

    n = len(trainingSet)
    n_five_percent = int(0.05 * n)

    index_of_selected_dataset = random.sample(range(n), n_five_percent)

    selected_dataset = np.copy(trainingSet[index_of_selected_dataset, :])

    # Deleting the selected data points from training dataset

    residual_training_dataset = np.delete(trainingSet, index_of_selected_dataset, 0)

    # print(selected_dataset.shape, dataset.shape)

    while len(selected_dataset) <= n / 2:
        training_x = selected_dataset[:, 0:57]
        training_y = selected_dataset[:, -1]

        classifier = adaboost_algo(selected_dataset, training_y, X_test, y_test, number_iterations)

        # Mutating list index_of_selected_dataset here

        indices_to_be_considered = predict_and_fetch_threshold_datapoints(classifier, residual_training_dataset)

        dataset_to_be_added = residual_training_dataset[indices_to_be_considered, :]
        selected_dataset = np.row_stack((selected_dataset, dataset_to_be_added))
        residual_training_dataset = np.delete(residual_training_dataset, indices_to_be_considered, 0)

    # Training accuracy and error
    prediction_y_train = predict(classifier, X_train)
    training_accuracy = evaluate_prediction_accuracy(y_train, prediction_y_train)

    print("Training accuracy is:", training_accuracy)
    print("Training error rate is:", 1 - training_accuracy)

    # Testing accuracy and error
    prediction_y_test = predict(classifier, X_test)
    testing_accuracy = evaluate_prediction_accuracy(y_test, prediction_y_test)

    print("Testing accuracy is:", testing_accuracy)
    print("Testing error rate is:", 1 - testing_accuracy)

    # Overall accuracy of dataset

    prediction_y_overall = predict(classifier, X)
    overall_accuracy = evaluate_prediction_accuracy(y, prediction_y_overall)

    print("Overall accuracy is:", overall_accuracy)
    print("Overall error rate is:", 1 - overall_accuracy)


if __name__ == '__main__':
    main()
