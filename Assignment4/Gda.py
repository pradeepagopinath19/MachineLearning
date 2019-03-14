import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



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


def count(number, y):
    total = 0
    for val in y:
        if val == number:
            total += 1
    return total


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100


def main():
    spam_dataset = extract_full_dataset()
    spam_dataset = shuffle(spam_dataset)
    #print(spam_dataset)
    dataset_k_split = kfold_split(4)

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


        mean_features_zero = []
        mean_features_one = []

        count_zero = count(0.0, training_y)
        count_one = count(1.0, training_y)

        prob_zero_y = count_zero / len(training_y)
        prob_one_y = count_one / len(training_y)

        for col in range(len(training_x[0])):
            sum_zero = []
            sum_one = []
            for row in range(len(training_x)):
                y = training_y[row]
                val = training_x[row][col]
                if y == 0.0:
                    sum_zero.append(val)
                else:
                    sum_one.append(val)

            mean_features_zero.append(np.mean(sum_zero) if len(sum_zero) > 0 else 0)
            mean_features_one.append(np.mean(sum_one) if len(sum_one) > 0 else 0)

        sigma = np.cov(training_x, rowvar=0)
        det_sigma = np.linalg.det(sigma)

        prediction_training = []
        for x in training_x:

            prob_x_y_zero = calculate_prob(det_sigma, sigma, x, mean_features_zero)
            prob_zero = prob_x_y_zero * prob_zero_y

            prob_x_y_one = calculate_prob(det_sigma, sigma, x, mean_features_one)
            prob_one = prob_x_y_one * prob_one_y

            # print(prob_zero, prob_one)
            if prob_zero >= prob_one:
                prediction_training.append(0.0)
            else:
                prediction_training.append(1.0)

        accuracy = calculate_accuracy(prediction_training, training_y)
        print('Training accuracy is', accuracy)
        full_accuracy_training.append(accuracy)

        prediction_testing= []
        for x in testing_x:

            prob_x_y_zero = calculate_prob(det_sigma, sigma, x, mean_features_zero)
            prob_zero = prob_x_y_zero * prob_zero_y

            prob_x_y_one = calculate_prob(det_sigma, sigma, x, mean_features_one)
            prob_one = prob_x_y_one * prob_one_y

            # print(prob_zero, prob_one)
            if prob_zero >= prob_one:
                prediction_testing.append(0.0)
            else:
                prediction_testing.append(1.0)

        accuracy = calculate_accuracy(prediction_testing, testing_y)
        print('Testing accuracy is', accuracy)
        full_accuracy_testing.append(accuracy)

    print("Training accuracy here is", full_accuracy_training)
    print("Overall training accuracy is", np.mean(full_accuracy_training))
    print("Testing accuracy here is", full_accuracy_testing)
    print("Overall testing accuracy is", np.mean(full_accuracy_testing))


def calculate_prob(det_sigma, sigma, x, mean):
    sub = np.subtract(x, mean).reshape(-1,1)

    mean_sqr = sub.T.dot(np.linalg.pinv(sigma)).dot(sub)

    return (1 / (math.sqrt((math.pow(2 * np.pi, 57)) * det_sigma))) * np.exp(-0.5 * mean_sqr)


if __name__ == '__main__':
    main()
