import numpy as np
import pandas as pd
import operator
from scipy import spatial
from sklearn import preprocessing


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def extract_haar_features():
    training_dataset = pd.read_csv("Haar_feature_full_training.csv", header=None, sep=',')
    testing_dataset = pd.read_csv("Haar_feature_testing.csv", header=None, sep=',')
    return training_dataset.values, testing_dataset.values


def fetch_labels_prior_count(trainingSet):
    local_dict = {}
    for row in trainingSet:
        label = row[-1]
        if label in local_dict:
            local_dict[label] += 1
        else:
            local_dict[label] = 1
    return local_dict


def calculate_class_priors(count_labels, n):
    local_dict = {}

    for key, value in count_labels.items():
        local_dict[key] = float(value / n)

    return local_dict


def calculate_poly_kernel_distance(x, xi):
    return (np.inner(x, xi) + 1) ** 2


def calculate_poly_prob(testingInstance, label_data_points):
    local_dict = {}

    for key, value in label_data_points.items():
        m_c = len(value)

        prob = 0

        for row in value:
            prob += calculate_poly_kernel_distance(testingInstance, row[0:-1])

        local_dict[key] = (1.0 / m_c) * prob
    return local_dict


def calculate_gaussian_kernel_distance(xi, x):
    sigma = 1.0
    return np.exp(-(spatial.distance.euclidean(xi, x) ** 2) / (2 * (sigma ** 2)))


def calculate_gaussian_prob(testingInstance, label_data_points):
    local_dict = {}

    for key, value in label_data_points.items():
        m_c = len(value)

        prob = 0

        for row in value:
            prob += calculate_gaussian_kernel_distance(testingInstance, row[0:-1])

        local_dict[key] = (1.0 / m_c) * prob
    return local_dict


def multiple_and_fetch_final_probability(prob_z_number, probabilty_prior):
    local_dict = {}

    for key in prob_z_number.keys():
        local_dict[key] = prob_z_number[key] * probabilty_prior[key]
    return local_dict


def main():
    trainingSet, testingSet = extract_haar_features()

    label_count = {5: 5421, 0: 5923, 4: 5842, 1: 6742, 9: 5949, 2: 5958, 3: 6131, 6: 5918, 7: 6265, 8: 5851}

    num_label_local = {}

    new_trainingset = []
    for row in trainingSet:
        label = row[-1]
        if label in num_label_local:
            if num_label_local[label] >= label_count[label] * 0.2:
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

    training_x = trainingSet[:, 0:-1]
    training_y = trainingSet[:, -1]

    testing_x = testingSet[:, 0:-1]
    testing_y = testingSet[:, -1]

    n_test = len(testingSet)
    n_train = len(trainingSet)

    count_labels = fetch_labels_prior_count(trainingSet)
    print(count_labels)

    probabilty_prior = calculate_class_priors(count_labels, n_train)
    print(probabilty_prior)

    label_data_points = {}

    for row in trainingSet:
        label = row[-1]
        if label not in label_data_points:
            label_data_points[label] = [row]
        else:
            label_data_points[label].append(row)

    # print(label_data_points)

    y_predications = []
    #n_test = 10
    for i in range(n_test):
        print(i, "done")
        # prob_z_number = calculate_gaussian_prob(testing_x[i], label_data_points)

        prob_z_number = calculate_poly_prob(testing_x[i], label_data_points)
        prob_number_z = multiple_and_fetch_final_probability(prob_z_number, probabilty_prior)

        # sorting and picking the number with the highest value

        sorted_prediction = sorted(prob_number_z.items(), key=operator.itemgetter(1), reverse=True)
        y_predications.append(sorted_prediction[0][0])

    print(len(y_predications), len(testing_y))
    print("Prediction", y_predications)
    print("True values", list(testing_y))
    accuracy = evaluate_prediction_accuracy(y_predications, testing_y[0:n_test])
    print("Testing accuracy is", accuracy)


if __name__ == '__main__':
    main()
