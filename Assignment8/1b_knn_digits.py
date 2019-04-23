import numpy as np
import pandas as pd
import random
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange
import matplotlib.pyplot as plt
import operator
from scipy import spatial
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import distance


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def extract_haar_features():
    # training_dataset = pd.read_csv("Haar_feature_training.csv", header=None, sep=',')
    training_dataset = pd.read_csv("Haar_feature_full_training.csv", header=None, sep=',')
    testing_dataset = pd.read_csv("Haar_feature_testing.csv", header=None, sep=',')
    return training_dataset.values, testing_dataset.values


def calculate_poly_kernel_distance(x, xi):
    return (np.inner(x, xi) + 1) ** 2


def fetch_neighbors_poly(trainingSet, testingInstance, k):
    instance_dist = {}

    for x in range(len(trainingSet)):
        training_x = trainingSet[x]
        dist = calculate_poly_kernel_distance(training_x[0:-1], testingInstance[0:-1])
        instance_dist[x] = dist

    sorted_nearest_neighbors = sorted(instance_dist.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_nearest_neighbors[:k]


def calculate_gaussian_kernel_distance(xi, x):
    sigma = 1.0
    return np.exp(-(spatial.distance.euclidean(xi, x) ** 2) / (2 * (sigma ** 2)))


def fetch_neighbors_gaussian(trainingSet, testingInstance, k):
    instance_dist = {}

    for x in range(len(trainingSet)):
        training_x = trainingSet[x]
        dist = calculate_gaussian_kernel_distance(training_x[0:-1], testingInstance[0:-1])
        instance_dist[x] = dist

    sorted_nearest_neighbors = sorted(instance_dist.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_nearest_neighbors[:k]


def fetch_neighbors_cosine(trainingSet, testingInstance, k):
    instance_dist = {}

    for x in range(len(trainingSet)):
        training_x = trainingSet[x]
        dist = spatial.distance.cosine(training_x[0:-1], testingInstance[0:-1])
        instance_dist[x] = dist

    sorted_nearest_neighbors = sorted(instance_dist.items(), key=operator.itemgetter(1))
    return sorted_nearest_neighbors[:k]


def get_Best_instances(k_neighbors_list, trainingSet):
    training_closest_instances = []
    for val in k_neighbors_list:
        training_closest_instances.append(trainingSet[val[0]])
    return training_closest_instances


def compute_best_prediction(k_neighbors):
    label_count = {}
    for row in k_neighbors:
        if row[-1] in label_count:
            label_count[row[-1]] += 1
        else:
            label_count[row[-1]] = 1
    best_prediction = sorted(label_count.items(), key=operator.itemgetter(1))
    return best_prediction[0][0]


def main():
    k = 7
    trainingSet, testingSet = extract_haar_features()

    label_count = {5: 5421, 0: 5923, 4: 5842, 1: 6742, 9: 5949, 2: 5958, 3: 6131, 6: 5918, 7: 6265, 8: 5851}

    num_label_local = {}

    new_trainingset = []
    for row in trainingSet:
        label = row[-1]
        if label in num_label_local:
            if num_label_local[label] >= label_count[label] * 0.5:
                continue
            num_label_local[label] += 1
            new_trainingset.append(row)
        else:
            num_label_local[label] = 1
            new_trainingset.append(row)

    trainingSet = np.array(new_trainingset)

    scaler = preprocessing.StandardScaler()
    scaler.fit(trainingSet[:, 0:-1])
    trainingSet[:, 0:-1] = scaler.transform(trainingSet[:, 0:-1])

    scaler = preprocessing.StandardScaler()
    scaler.fit(testingSet[:, 0:-1])
    testingSet[:, 0:-1] = scaler.transform(testingSet[:, 0:-1])

    y_predications = []
    iter = 0

    n = len(testingSet)
    #n = 50
    for row in range(n):
        iter += 1
        print(iter)

        # Cosine distance
        # k_neighbors = fetch_neighbors_cosine(trainingSet, testingSet[row], k)

        # Gaussian Kernel distance
        k_neighbors = fetch_neighbors_gaussian(trainingSet, testingSet[row], k)

        # Poly kernel distance
        # k_neighbors = fetch_neighbors_poly(trainingSet, testingSet[row], k)

        instances_best_neighbors = get_Best_instances(k_neighbors, trainingSet)
        label_best_prediction = compute_best_prediction(instances_best_neighbors)
        y_predications.append(label_best_prediction)
    true_labels = testingSet[:, -1]

    print(len(y_predications), len(true_labels))
    print("Prediction", y_predications)
    print("True values", list(true_labels))
    accuracy = evaluate_prediction_accuracy(y_predications, true_labels[:n])
    print("Testing accuracy is", accuracy)


if __name__ == '__main__':
    main()
