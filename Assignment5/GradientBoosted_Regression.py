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
import operator

from RegressionTree import build_tree, fetch_dataset, test_model, evaluate_prediction


def add_two_lists(a, b):
    if len(a) == 0:
        return b
    return [a[i] + b[i] for i in range(len(a))]


def main_child():
    training_dataset, testing_dataset = fetch_dataset()

    # Deep copy to save the original copy
    unchanged_dataset = np.copy(training_dataset)
    num_of_iterations = 10
    depth = 2
    stopping_size = 10

    regression_trees = []

    # Building model
    for i in range(num_of_iterations):
        tree = build_tree(training_dataset, depth, stopping_size)
        predictedValues = test_model(training_dataset, tree)
        training_dataset[:, -1] -= predictedValues
        regression_trees.append(tree)

    # Prediction
    predictedValues = []
    for tree in regression_trees:
        predictedValues = add_two_lists(predictedValues, test_model(unchanged_dataset, tree))

    print(len(predictedValues))
    print(len(unchanged_dataset[:,-1]))
    evaluateModel_mse_training = evaluate_prediction(predictedValues, training_dataset)
    print("The calculated MSE for training dataset is", evaluateModel_mse_training)

    predictedValues = []
    for tree in regression_trees:
        predictedValues = add_two_lists(predictedValues, test_model(testing_dataset, tree))

    evaluateModel_mse_testing = evaluate_prediction(predictedValues, testing_dataset)
    print("The calculated MSE for testing dataset is", evaluateModel_mse_testing)


if __name__ == '__main__':
    main_child()
