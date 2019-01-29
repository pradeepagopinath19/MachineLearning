import numpy as np
import pandas as pd
import random
import math
from random import shuffle

def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    training_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt"
    testing_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt"
    training_data = pd.read_csv(training_url, header=None, sep='\s+')
    testing_data = pd.read_csv(testing_url, header=None, sep='\s+')
    return spam_dataset, training_data.values, testing_data.values

def evaluate_prediction(estimation, trueValues):
    errorValues = np.array(estimation) - np.array(trueValues)
    return (np.sum(np.square(errorValues)) / len(errorValues))

def kfold_split(k):
    kfold_list = random.sample(range(0, k), k)
    return kfold_list

def get_training_testing_split(dataset, split, index):
    k = len(split)
    len_of_k = len(dataset) // k
    starting_row = index * len_of_k
    ending_row = starting_row + len_of_k
    # print(starting_row, ending_row)
    testing_data = dataset.iloc[starting_row:ending_row, :]
    training_data1 = dataset.iloc[0:starting_row, :]
    training_data2 = dataset.iloc[ending_row:len(dataset), :]
    training_data = training_data1.append(training_data2, sort=False)
    # print(testing_data)

    return training_data, testing_data

def evaluate_prediction_accuracy(predictedValues, actualValues, classification_threshold):
    normalized_prediction =[]
    for i in predictedValues:
        if i >= classification_threshold:
            normalized_prediction.append(1)
        else:
            normalized_prediction.append(0)
    correct_predictions = [i for i, j in zip(normalized_prediction, actualValues) if i == j]
    return float(len(correct_predictions)) / len(actualValues) * 100

def main():
    spam_dataset, housing_training, housing_testing = extract_full_dataset()

    # Housing section

    Y_test =[row[-1] for row in housing_testing]
    X_test =housing_testing[:,0:len(housing_testing[0])-1]

    Y =[row[-1] for row in housing_training]
    X = housing_training[:,0:len(housing_training[0])-1]

    # Adding one to each instance
    X_b = np.c_[np.ones((len(X), 1)), X]
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

    # Adding one to each instance
    X_new_b = np.c_[np.ones((len(X_test), 1)), X_test]
    y_predict = X_new_b.dot(w)
    print("MSE for housing is:",evaluate_prediction(y_predict,Y_test))


    # Spam section

    dataset_k_split = kfold_split(5)
    #print(dataset_k_split)


    spam_accuracy =[]
    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(spam_dataset, dataset_k_split, i)
        trainingSet = trainingSet.values
        testingSet = testingSet.values

        #trainingSet = np.random.shuffle(trainingSet)
        print(trainingSet)
        Y_test = [row[-1] for row in testingSet]
        X_test = testingSet[:, 0:len(testingSet[0]) - 1]

        Y = [row[-1] for row in trainingSet]
        X = trainingSet[:, 0:len(trainingSet[0]) - 1]

        # Adding one to each instance
        X_b = np.c_[np.ones((len(X), 1)), X]
        w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

        # Adding one to each instance
        X_new_b = np.c_[np.ones((len(X_test), 1)), X_test]
        y_predict = X_new_b.dot(w)
        #print("Prediction is:", y_predict)
        #print(Y_test)

        accuracy = evaluate_prediction_accuracy(y_predict, Y_test, classification_threshold=0.38)

        spam_accuracy.append(accuracy)

    print("Individual run accuracy list:", spam_accuracy)
    print("Mean of accuracy is:", np.mean(spam_accuracy))

if __name__ == '__main__':
    main()


