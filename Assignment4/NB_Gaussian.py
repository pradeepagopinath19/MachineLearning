import numpy as np
import pandas as pd
import random
import math
import ConfusionMatrix
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
    #print(output)
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
    spam_dataset = extract_full_dataset()
    spam_dataset = shuffle(spam_dataset, random_state=0)
    dataset_k_split = kfold_split(2)

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

        training_collection = build_dictionary(trainingSet)

        statistics = {}

        for classValue, instances in training_collection.items():
            statistics[classValue] = summarize(instances)

        # Calculating training and testing error
        train_prediction, train_prediction_prob = predict(training_x, statistics, prob_zero_y, prob_one_y)
        test_prediction, test_prediction_prob = predict(testing_x, statistics, prob_zero_y, prob_one_y)

        # Confusion Matrix, ROC, AUC
        #print(test_prediction_prob)


        ConfusionMatrix.confusion_matrix(testing_y, test_prediction, test_prediction_prob, True)

        train_accuracy = calculate_accuracy(train_prediction, training_y)
        test_accuracy = calculate_accuracy(test_prediction, testing_y)

        full_accuracy_training.append(train_accuracy)
        full_accuracy_testing.append(test_accuracy)

    print("Individual training accuracy using Gaussian random variable is ", full_accuracy_training)
    print("Mean value of Gaussian random variable training is", np.mean(full_accuracy_training))

    print("Individual testing accuracy using Gaussian random variable is ", full_accuracy_testing)
    print("Mean value of Gaussian random variable testing is", np.mean(full_accuracy_testing))


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, variance = classSummaries[i]
            x = inputVector[i]
            #print(x,mean,variance)
            probabilities[classValue] *= calculate_prob(x, mean, variance)
    return probabilities


if __name__ == '__main__':
    main()
