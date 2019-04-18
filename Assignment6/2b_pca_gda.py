import pandas as pd
import numpy as np
import numbers
import re
import random
import math
from sklearn.utils import shuffle
from sklearn import decomposition


def extract_full_dataset():
    training_features = pd.read_csv("spam_polluted_no_missing/train_feature.txt", header=None, sep='\s+').values
    training_label = pd.read_csv("spam_polluted_no_missing/train_label.txt", header=None, sep='\s+').values
    testing_features = pd.read_csv("spam_polluted_no_missing/test_feature.txt", header=None, sep='\s+').values
    testing_label = pd.read_csv("spam_polluted_no_missing/test_label.txt", header=None, sep='\s+').values

    training_dataset = np.column_stack((training_features, training_label))
    testing_dataset = np.column_stack((testing_features, testing_label))

    return training_dataset, testing_dataset


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100


def summarize(dataset):
    summaries = [(np.mean(attribute), np.var(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def build_dictionary(dataset):
    dictionary = {}
    for j in range(len(dataset)):
        data_point = dataset[j]
        class_value = data_point[-1]
        if class_value not in dictionary:
            dictionary[class_value] = []
        dictionary[class_value].append(data_point)
    return dictionary


def count(number, y):
    total = 0
    for val in y:
        if val == number:
            total += 1
    return total


def calculate_prob(x, mean, variance):
    if variance <= 0.001:
        variance = 0.001
    loss = (np.power(x - mean, 2)) / variance

    exp_val = math.exp(-0.5 * loss)

    output = (1 / (math.sqrt(2 * math.pi * variance))) * exp_val
    # print(output)
    return output


def predict(x, statistics, prob_y_zero, prob_y_one):
    predictions = []
    predictions_probability = []
    for i in range(len(x)):
        result = calculateClassProbabilities(statistics, x[i])
        result[0.0] *= prob_y_zero
        result[1.0] *= prob_y_one
        bestLabel, bestProb = None, -1
        for classValue, probability in result.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        predictions_probability.append(bestProb)
        predictions.append(bestLabel)
    return predictions, predictions_probability


def main():
    trainingSet, testingSet = extract_full_dataset()

    trainingSet = shuffle(trainingSet)

    training_x = np.copy(trainingSet[:, 0:-1])
    training_y = np.copy(trainingSet[:, -1])

    testing_x = np.copy(testingSet[:, 0:-1])
    testing_y = np.copy(testingSet[:, -1])

    # PCA

    pca = decomposition.PCA(n_components=100)
    pca.fit(training_x)
    training_x = pca.transform(training_x)

    testing_x = pca.transform(testing_x)

    print(training_x.shape, testing_x.shape)

    # Updating training set and testing set after PCA
    trainingSet = np.column_stack((training_x, training_y))
    testingSet = np.column_stack((testing_x, testing_y))
    # Class priors - Training

    count_zero = count(0.0, training_y)
    count_one = count(1.0, training_y)

    prob_zero_y = count_zero / len(training_y)
    prob_one_y = count_one / len(training_y)

    # print(prob_zero_y, prob_one_y)

    training_collection = build_dictionary(trainingSet)

    statistics = {}

    for classValue, instances in training_collection.items():
        statistics[classValue] = summarize(instances)

    # Calculating training and testing error
    train_prediction, train_prediction_prob = predict(training_x, statistics, prob_zero_y, prob_one_y)
    test_prediction, test_prediction_prob = predict(testing_x, statistics, prob_zero_y, prob_one_y)

    train_accuracy = calculate_accuracy(train_prediction, training_y)
    test_accuracy = calculate_accuracy(test_prediction, testing_y)

    print("Training accuracy:", train_accuracy)

    print("Testing accuracy is:", test_accuracy)


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, variance = classSummaries[i]
            x = inputVector[i]
            # print(x,mean,variance)
            probabilities[classValue] *= calculate_prob(x, mean, variance)
    return probabilities


if __name__ == '__main__':
    main()
