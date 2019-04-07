import numpy as np
import pandas as pd
import random
from DecisionStump import DecisionStump
import math
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


def make_prediction_from_stump(dataset, feature_number, threshold_val):
    prediction = []
    for row in dataset:
        if row[feature_number] < threshold_val:
            prediction.append(-1)
        else:
            prediction.append(1)

    return prediction


def adaboost_algo(dataset, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones((len(dataset), 1)) / len(dataset)

    dec_classifiers = []

    for _ in range(max_iter):

        classifier = DecisionStump()

        # Best decision stump
        for j in range(len(dataset[0]) - 1):
            dataset = dataset[dataset[:, j].argsort()]
            y = dataset[:, -1]

            min_weighted_error = math.inf
            previous_threshold = -math.inf
            best_prediction = []

            for i in range(len(dataset)):
                if previous_threshold == dataset[i][j]:
                    continue
                previous_threshold = dataset[i][j]

                stump_prediction = make_prediction_from_stump(dataset, j, previous_threshold)

                index = 0
                weighted_error = 0
                for actual, prediction in zip(y, stump_prediction):
                    if actual != prediction:
                        weighted_error += w[index]
                    index += 1

                if weighted_error > 0.5:
                    p = -1
                    weighted_error = 1 - weighted_error
                else:
                    p = 1

                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error
                    best_prediction = stump_prediction

                    classifier.threshold = previous_threshold
                    classifier.feature = j
                    classifier.polarity = p
            classifier.alpha = 0.5 * math.log((1.0 - min_weighted_error) / (min_weighted_error + 1e-12))

            if classifier.polarity == -1:
                prediction = np.multiply(best_prediction, -1)

            else:
                prediction = best_prediction

            prediction = np.array(prediction).reshape(len(prediction), 1)

            y = np.array(y).reshape(len(y), 1)
            # Updating w

            w *= np.exp(classifier.alpha * y * prediction)

            w /= np.sum(w)

            dec_classifiers.append(classifier)
            print('Done')
    return dec_classifiers


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def predict(classifiers, X):
    y_pred = np.zeros((len(X), 1))

    for c in classifiers:
        non_spam_idx = (c.polarity * X[:, c.feature] < c.polarity * c.threshold)
        #print(non_spam_idx)

        predictions = np.ones((len(X), 1))
        predictions[non_spam_idx] = -1
        y_pred += c.alpha * predictions

    return np.sign(y_pred).flatten()



def main():
    dataset = extract_full_dataset()
    # print(dataset)

    spam_dataset = shuffle(dataset)

    dataset_k_split = kfold_split(5)
    number_iterations = 100

    full_accuracy_testing = []

    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(spam_dataset, dataset_k_split, i)

        trainingSet = trainingSet.values
        testingSet = testingSet.values

        # {1,-1}

        training_y_col = len(trainingSet[0]) - 1

        for row_no_training in range(len(trainingSet)):
            if trainingSet[row_no_training][training_y_col] == 0:
                trainingSet[row_no_training][training_y_col] = -1

        testing_y_col = len(testingSet[0]) - 1
        for row_no_testing in range(len(testingSet)):
            if testingSet[row_no_testing][testing_y_col] == 0:
                testingSet[row_no_testing][testing_y_col] = -1

        training_x = trainingSet[:, 0:57]
        training_y = trainingSet[:, -1]

        testing_x = testingSet[:, 0:57]
        testing_y = testingSet[:, -1]

        classifiers = adaboost_algo(trainingSet, number_iterations)
        prediction_y = predict(classifiers, testing_x)

        full_accuracy_testing.append(evaluate_prediction_accuracy(testing_y, prediction_y))

    print("Individual run accuracy list:", full_accuracy_testing)
    print("Mean of accuracy is:", np.mean(full_accuracy_testing))


if __name__ == '__main__':
    main()
