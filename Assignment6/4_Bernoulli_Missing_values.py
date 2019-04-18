import pandas as pd
import numpy as np
import random


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
    summaries = [np.nanmean(attribute) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def build_mean_dictionary(dict):
    statistics = {}
    for classValue, instances in dict.items():
        statistics[classValue] = summarize(instances)
    return statistics


def calculate_bernoulli_freq(label, feature_num, mean_value, training_collection, greater_than):
    count = 0
    for instances in training_collection[label]:
        if greater_than:
            if instances[feature_num] > mean_value:
                count += 1
        else:
            if instances[feature_num] <= mean_value:
                count += 1
    return (count + 1) / (len(training_collection[label]) + 2)
    # return (count + 2) / (len(training_collection[label]) + 4)


def build_bernoulli_dictionary(training_collection, mean_dict):
    bernoulli = {}
    bernoulli[0.0] = {}
    bernoulli[1.0] = {}
    # print(training_collection)
    for i in range(57):
        nonspam_lessThanMean = calculate_bernoulli_freq(0.0, i, mean_dict[0.0][i], training_collection, False)
        nonspam_greaterThanMean = calculate_bernoulli_freq(0.0, i, mean_dict[0.0][i], training_collection, True)
        spam_lessThanMean = calculate_bernoulli_freq(1.0, i, mean_dict[1.0][i], training_collection, False)
        spam_greaterThanMean = calculate_bernoulli_freq(1.0, i, mean_dict[1.0][i], training_collection, True)
        bernoulli[0.0][i] = {0.0: nonspam_lessThanMean, 1.0: nonspam_greaterThanMean}
        bernoulli[1.0][i] = {0.0: spam_lessThanMean, 1.0: spam_greaterThanMean}
    return bernoulli


def calculateBernProbabilities(bern_dict, mean_dict, data_point):
    result = {}
    for classValue in bern_dict.keys():
        result[classValue] = 1
        for i, x in enumerate(data_point):
            result[classValue] *= bern_dict[classValue][i][0.0] if x <= mean_dict[classValue][i] \
                else bern_dict[classValue][i][1.0]
    return result


def predict(x, mean_dict, bern_dict, prob_y_zero, prob_y_one):
    predictions = []
    for i in range(len(x)):
        result = calculateBernProbabilities(bern_dict, mean_dict, x[i])
        result[0.0] *= prob_y_zero
        result[1.0] *= prob_y_one
        bestLabel, bestProb = None, -1
        for classValue, probability in result.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        predictions.append(bestLabel)
    return predictions


def extract_full_dataset():
    training_dataset = pd.read_csv(
        "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/spam_missing_values/20_percent_missing_train.txt",
        header=None, sep=',')

    testing_dataset = pd.read_csv(
        "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/spam_missing_values/20_percent_missing_test.txt",
        header=None, sep=',')

    return training_dataset.values, testing_dataset.values


def main():
    # Fetching dataset
    trainingSet, testingSet = extract_full_dataset()

    # shuffle
    # trainingSet = shuffle(trainingSet)

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

    # Mean for each feature - dictionary

    mean_dict = build_mean_dictionary(training_collection)
    # print(mean_dict[0.0])

    # Bernoulli dictionary
    bern_dict = build_bernoulli_dictionary(training_collection, mean_dict)

    train_accuracy = calculate_accuracy(predict(training_x, mean_dict, bern_dict, prob_zero_y, prob_one_y),
                                        training_y)
    test_accuracy = calculate_accuracy(predict(testing_x, mean_dict, bern_dict, prob_zero_y, prob_one_y), testing_y)

    # print("Training accuracy is", np.mean(train_accuracy))

    print("Testing accuracy is", test_accuracy)


if __name__ == '__main__':
    main()
