# Citation - https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
import numpy as np
import pandas as pd
import random
import math
from sklearn.utils import shuffle


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100


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


def summarize(dataset):
    summaries = [(np.min(attribute), np.mean(attribute), np.max(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def get_overall_mean(instances):
    means = [np.mean(attribute) for attribute in zip(*instances)]
    return means


def build_hist_values(dict, training_x):
    statistics = {}

    for classValue, instances in dict.items():
        statistics[classValue] = summarize(instances)

    overall_mean = get_overall_mean(training_x)
    updated_statistics = {}
    updated_statistics[0.0] = []
    updated_statistics[1.0] = []
    count = 0
    for tuple1, tuple2 in zip(statistics[0.0], statistics[1.0]):
        low_mean = min(tuple1[1], tuple2[1])
        high_mean = max(tuple1[1], tuple2[1])
        overall_min = min(tuple1[0], tuple2[0])
        overall_max = max(tuple1[2], tuple2[2])
        updated_statistics[0.0].append((overall_min, low_mean, overall_mean[count], high_mean, overall_max))
        updated_statistics[1.0].append((overall_min, low_mean, overall_mean[count], high_mean, overall_max))
        count += 0
    return updated_statistics


def calculate_bin_freq(label, feature_num, hist_points, training_collection, bin_number):
    count = 0
    for instances in training_collection[label]:
        if bin_number == 0:
            if instances[feature_num] >= hist_points[0] and instances[feature_num] < hist_points[1]:
                count += 1
        elif bin_number == 1:
            if instances[feature_num] >= hist_points[1] and instances[feature_num] < hist_points[2]:
                count += 1
        elif bin_number == 2:
            if instances[feature_num] >= hist_points[2] and instances[feature_num] < hist_points[3]:
                count += 1
        elif bin_number == 3:
            if instances[feature_num] >= hist_points[3] and instances[feature_num] < hist_points[4]:
                count += 1
        else:
            count = 0

    return (count + 1) / (len(training_collection[label]) + 2)


def build_bin_dictionary(training_collection, histogram_points_dict):
    bin_dict = {}
    bin_dict[0.0] = {}
    bin_dict[1.0] = {}

    for i in range(57):
        nonspam_0 = calculate_bin_freq(0.0, i, histogram_points_dict[0.0][i], training_collection, 0)
        nonspam_1 = calculate_bin_freq(0.0, i, histogram_points_dict[0.0][i], training_collection, 1)
        nonspam_2 = calculate_bin_freq(0.0, i, histogram_points_dict[0.0][i], training_collection, 2)
        nonspam_3 = calculate_bin_freq(0.0, i, histogram_points_dict[0.0][i], training_collection, 3)
        spam_0 = calculate_bin_freq(1.0, i, histogram_points_dict[1.0][i], training_collection, 0)
        spam_1 = calculate_bin_freq(1.0, i, histogram_points_dict[1.0][i], training_collection, 1)
        spam_2 = calculate_bin_freq(1.0, i, histogram_points_dict[1.0][i], training_collection, 2)
        spam_3 = calculate_bin_freq(1.0, i, histogram_points_dict[1.0][i], training_collection, 3)

        bin_dict[0.0][i] = {0.0: nonspam_0, 1.0: nonspam_1, 2.0: nonspam_2, 3.0: nonspam_3}
        bin_dict[1.0][i] = {0.0: spam_0, 1.0: spam_1, 2.0: spam_2, 3.0: spam_3}
    # print(bin_dict)
    # exit()
    return bin_dict


def fetch_value_from_histogram(bin_dict, hist_dict, classValue, i, x, count_zero, count_one):
    count = count_zero if classValue == 0.0 else count_one
    if x >= hist_dict[classValue][i][0] and x < hist_dict[classValue][i][1]:
        return bin_dict[classValue][i][0.0]
    elif x >= hist_dict[classValue][i][1] and x < hist_dict[classValue][i][2]:
        return bin_dict[classValue][i][1.0]
    elif x >= hist_dict[classValue][i][2] and x < hist_dict[classValue][i][3]:
        return bin_dict[classValue][i][2.0]
    elif x >= hist_dict[classValue][i][3] and x <= hist_dict[classValue][i][4]:
        return bin_dict[classValue][i][3.0]
    else:
        return 1 / (count + 2)


def calculateBinProbabilities(bin_dict, hist_dict, data_point, count_zero, count_one):
    result = {}
    for classValue in bin_dict.keys():
        result[classValue] = 1
        for i, x in enumerate(data_point):
            result[classValue] *= fetch_value_from_histogram(bin_dict, hist_dict, classValue, i, x, count_zero,
                                                             count_one)
    return result


def predict(x, hist_dict, bin_dict, prob_y_zero, prob_y_one, count_zero, count_one):
    predictions = []
    for i in range(len(x)):
        result = calculateBinProbabilities(bin_dict, hist_dict, x[i], count_zero, count_one)
        result[0.0] *= prob_y_zero
        result[1.0] *= prob_y_one
        bestLabel, bestProb = None, -1
        for classValue, probability in result.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        predictions.append(bestLabel)
    return predictions


def main():
    spam_dataset = extract_full_dataset()
    spam_dataset = shuffle(spam_dataset, random_state=0)

    dataset_k_split = kfold_split(10)

    full_accuracy_training = []
    full_accuracy_testing = []

    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(spam_dataset, dataset_k_split, i)

        trainingSet = trainingSet.values
        testingSet = testingSet.values

        training_x = trainingSet[:, 0:57]
        training_y = trainingSet[:, -1]

        testing_x = testingSet[:, 0:57]
        testing_y = testingSet[:, -1]

        # Class priors - Training

        count_zero = count(0.0, training_y)
        count_one = count(1.0, training_y)

        prob_zero_y = count_zero / len(training_y)
        prob_one_y = count_one / len(training_y)

        # print(prob_zero_y, prob_one_y)

        # Classification dictionary
        training_collection = build_dictionary(trainingSet)

        # print(training_collection[0.0])
        # print(training_collection[1.0])

        # calculate the points of the histogram
        histogram_points_dict = build_hist_values(training_collection, training_x)
        # print("Histogram_dict", histogram_points_dict)

        # build 4 bin dictionary

        bin_dict = build_bin_dictionary(training_collection, histogram_points_dict)

        train_accuracy = calculate_accuracy(
            predict(training_x, histogram_points_dict, bin_dict, prob_zero_y, prob_one_y, count_zero, count_one),
            training_y)
        test_accuracy = calculate_accuracy(
            predict(testing_x, histogram_points_dict, bin_dict, prob_zero_y, prob_one_y, count_zero, count_one),
            testing_y)

        full_accuracy_training.append(train_accuracy)
        full_accuracy_testing.append(test_accuracy)

    print("Individual training accuracy using Gaussian random variable is ", full_accuracy_training)
    print("Mean value of Gaussian random variable training is", np.mean(full_accuracy_training))

    print("Individual testing accuracy using Gaussian random variable is ", full_accuracy_testing)
    print("Mean value of Gaussian random variable testing is", np.mean(full_accuracy_testing))


if __name__ == '__main__':
    main()
