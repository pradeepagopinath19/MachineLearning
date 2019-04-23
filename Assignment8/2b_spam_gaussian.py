import numpy as np
import pandas as pd
import operator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
from sklearn import preprocessing
from scipy import spatial


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset


def count(number, y):
    total = 0
    for val in y:
        if val == number:
            total += 1
    return total


def calculate_gaussian_kernel_distance(xi, x):
    sigma = 1.0
    return np.exp(-(spatial.distance.euclidean(xi, x) ** 2) / (2 * (sigma ** 2)))


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def calculate_gaussian_prob(z, label_data_points, label_value):
    m_c = len(label_data_points[label_value])

    prob = 0
    for row in label_data_points[label_value]:
        prob += calculate_gaussian_kernel_distance(z, row[0:-1])
    return (1.0 / m_c) * prob


def main():
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

    training_x = trainingSet[:, 0:-1]
    training_y = trainingSet[:, -1]

    testing_x = testingSet[:, 0:-1]
    testing_y = testingSet[:, -1]

    count_zero = count(0.0, training_y)
    count_one = count(1.0, training_y)

    prob_zero_y = count_zero / len(training_y)
    prob_one_y = count_one / len(training_y)

    label_data_points = {}

    for row in trainingSet:
        label = row[-1]
        if label not in label_data_points:
            label_data_points[label] = [row]
        else:
            label_data_points[label].append(row)

    prediction = []
    true_labels = testing_y

    for z in testing_x:
        prob_z_zero = calculate_gaussian_prob(z, label_data_points, 0.0)
        prob_z_one = calculate_gaussian_prob(z, label_data_points, 1.0)

        prob_zero_z = prob_z_zero * prob_zero_y
        prob_one_z = prob_z_one * prob_one_y

        if prob_zero_z > prob_one_z:
            prediction.append(0)
        else:
            prediction.append(1)

    print(len(prediction), len(true_labels))
    print("Prediction", prediction)
    print("True values", list(true_labels))
    accuracy = evaluate_prediction_accuracy(prediction, true_labels)
    print("Testing accuracy is", accuracy)


if __name__ == '__main__':
    main()
