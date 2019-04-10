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
from DecisionTree import build_tree, test_model, evaluate_prediction_accuracy


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset.values


def get_final_prediction(predicted_values):
    return np.round(np.mean(predicted_values, axis=0))


def main():
    dataset = extract_full_dataset()
    dataset = shuffle(dataset)
    # print(dataset.shape)

    X = dataset[:, 0:57]
    y = dataset[:, -1]

    number_iterations = 50
    maximum_depth = 2
    stopping_size = 15

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    length_X_60 = int(0.3 * len(X_train))

    decision_trees = []

    # Bagging - Training
    for iteration in range(number_iterations):
        X_train_random_index = np.random.randint(length_X_60, size=length_X_60)
        training_dataset_bagging = trainingSet[X_train_random_index, :]
        dec_tree_model = build_tree(training_dataset_bagging, maximum_depth, stopping_size)
        decision_trees.append(dec_tree_model)

    # Prediction

    # Training
    predictedValues = []
    for tree in decision_trees:
        prediction = test_model(trainingSet, tree)
        predictedValues.append(prediction)
        print(prediction)

    final_prediction = get_final_prediction(predictedValues)
    actualValues = [row[-1] for row in trainingSet]
    accuracy = evaluate_prediction_accuracy(final_prediction, actualValues)
    print("Training accuracy is", accuracy)

    # Testing
    predictedValues = []
    for tree in decision_trees:
        prediction = test_model(testingSet, tree)
        predictedValues.append(prediction)
        print(prediction)

    final_prediction = get_final_prediction(predictedValues)
    actualValues = [row[-1] for row in testingSet]
    accuracy = evaluate_prediction_accuracy(final_prediction, actualValues)
    print("Testing accuracy is", accuracy)


if __name__ == '__main__':
    main()
