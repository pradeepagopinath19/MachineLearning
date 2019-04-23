import numpy as np
import pandas as pd
import operator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from sklearn import preprocessing


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def calculate_euclidean_distance(x1, x2):
    dist = 0
    for i in range(len(x1) - 1):
        dist += pow((x1[i] - x2[i]), 2)

    return math.sqrt(dist)


def fetch_neighbors(trainingSet, testingInstance, k):
    instance_dist = {}

    for x in range(len(trainingSet)):
        dist = calculate_euclidean_distance(trainingSet[x], testingInstance)
        instance_dist[x] = dist

    sorted_nearest_neighbors = sorted(instance_dist.items(), key=operator.itemgetter(1))
    print(sorted_nearest_neighbors[:k])
    return sorted_nearest_neighbors[:k]


def compute_best_prediction(k_neighbors):
    spam_count = 0
    nonspam_count = 0
    for row in k_neighbors:
        if row[-1] == 0:
            nonspam_count += 1
        else:
            spam_count += 1
    return 0 if nonspam_count >= spam_count else 1


def get_Best_instances(k_neighbors_list, trainingSet):
    training_closest_instances = []
    for val in k_neighbors_list:
        training_closest_instances.append(trainingSet[val[0]])
    return training_closest_instances


def main():
    k = 2

    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    # Pandas to numpy array
    dataset = dataset.values

    # print(dataset.shape)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    y_predications = []
    true_labels = testingSet[:, -1]

    for row in range(len(testingSet)):
        k_neighbors = fetch_neighbors(trainingSet, testingSet[row], k)
        instances_best_neighbors = get_Best_instances(k_neighbors, trainingSet)
        label_best_prediction = compute_best_prediction(instances_best_neighbors)
        y_predications.append(label_best_prediction)

    print(len(y_predications), len(true_labels))
    accuracy = evaluate_prediction_accuracy(y_predications, true_labels)
    print("Testing accuracy is", accuracy)


if __name__ == '__main__':
    main()
