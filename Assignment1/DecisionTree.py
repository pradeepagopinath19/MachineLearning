import random
import numpy as np
import math
import random
import pandas as pd
import time

def split_data(dataset, featureColumn, thresholdRow):
    thresholdValue = dataset[thresholdRow][featureColumn]
    # print(thresholdValue)
    # print(featureColumnValues)

    left, right = [], []
    for row in dataset:
        if row[featureColumn] <= thresholdValue:
            left.append(row)
        else:
            right.append(row)
    return left, right

def extract_full_dataset():
    dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    dataset = pd.read_csv(dataset_url, header=None, sep=',')
    return dataset


def kfold_split(k):
    kfold_list = random.sample(range(0, k), k)
    return kfold_list


def get_training_testing_split(dataset, split, index):
    k = len(split)
    len_of_k = math.floor(float(len(dataset) / k))
    starting_row = index * len_of_k
    ending_row = starting_row + len_of_k
    # print(starting_row, ending_row)
    testing_data = dataset.iloc[starting_row:ending_row, :]
    training_data1 = dataset.iloc[0:starting_row, :]
    training_data2 = dataset.iloc[ending_row:len(dataset), :]
    training_data = training_data1.append(training_data2, sort=False)
    # print(testing_data)

    return training_data, testing_data


def leaf_node(data):
    featureValues = [row[-1] for row in data]

    count_of_zeroes = 0
    count_of_ones = 0
    for i in featureValues:
        if i == 0:
            count_of_zeroes += 1
        else:
            count_of_ones += 1

    return 1 if (count_of_zeroes < count_of_ones) else 0


def split_tree(node, max_depth, stopping_size, depth):
    print(node['feature'])
    print(node['threshold'])

    # stopping criteria

    if depth >= max_depth:
        node['llink'] = leaf_node(node['left'])
        node['rlink'] = leaf_node(node['right'])
        return

    if not node['left'] or not node['right']:
        node['llink'] = node['rlink'] = leaf_node(node['left'] + node['right'])
        return

    # recursion steps
    if len(node['left']) <= stopping_size:
        node['llink'] = leaf_node(node['left'])
    else:
        node['llink'] = find_best_split(node['left'])
        split_tree(node['llink'], max_depth, stopping_size, depth + 1)

    if len(node['right']) <= stopping_size:
        node['rlink'] = leaf_node(node['right'])
    else:
        node['rlink'] = find_best_split(node['right'])
        split_tree(node['rlink'], max_depth, stopping_size, depth + 1)


def entropy(dataList):
    countOfZeroes = 0
    countOfOnes = 0

    if len(dataList) == 0:
        return 0
    len_of_dataset = len(dataList)

    for i in dataList:
        if i == 0:
            countOfZeroes += 1
        elif i == 1:
            countOfOnes += 1
    probabilityOfZero = countOfZeroes / len_of_dataset
    if probabilityOfZero != 0:
        log_reciprocal_zero = math.log((1 / probabilityOfZero), 2)
    else:
        log_reciprocal_zero = 0

    probabilityOfOne = countOfOnes / len_of_dataset

    if probabilityOfOne != 0:
        log_reciprocal_one = math.log((1 / probabilityOfOne), 2)
    else:
        log_reciprocal_one = 0

    return (probabilityOfZero * log_reciprocal_zero) + (probabilityOfOne * log_reciprocal_one)


def calculate_entropy(left, right, len_of_label):
    leftLabelValues = [row[-1] for row in left]
    rightLabelValues = [row[-1] for row in right]

    left_entropy = entropy(leftLabelValues)
    right_entropy = entropy(rightLabelValues)

    return (left_entropy * len(leftLabelValues) / len_of_label) + (right_entropy * len(rightLabelValues) / len_of_label)


def find_best_split(dataset):
    len_of_label = len([row[-1] for row in dataset])
    minimumEntropy = math.inf
    minimumFeature = 0
    minimumThreshold = 0
    minimumLeft, minimumRight = [], []
    left, right = [], []
    previous = -math.inf
    dataset= np.array(dataset)
    for j in range(len(dataset[0]) - 1):
        dataset = dataset[dataset[: , j].argsort()]
        for i in range(len(dataset)):
            if previous == dataset[i][j]:
                continue
            previous = dataset[i][j]
            left, right = split_data(dataset, j, i)
            calculatedEntropy = calculate_entropy(left, right, len_of_label)
            if calculatedEntropy < minimumEntropy:
                minimumEntropy = calculatedEntropy
                minimumFeature = j
                minimumThreshold = dataset[i][j]
                minimumLeft = left
                minimumRight = right
    print("Feature: %d minimumThreshold: %f Error: %f" % (minimumFeature + 1, minimumThreshold, minimumEntropy))
    return {'left': minimumLeft, 'right': minimumRight, 'feature': minimumFeature, 'threshold': minimumThreshold,
            'error': minimumEntropy}


def build_tree(dataset, max_depth, stopping_size):
    root_node = find_best_split(dataset)
    split_tree(root_node, max_depth, stopping_size, 1)
    return root_node

def predict(tree_model, test_row):

    if test_row[tree_model['feature']] < tree_model['threshold']:
        if isinstance(tree_model['llink'], dict):
            return predict(tree_model['llink'], test_row)
        else:
            return tree_model['llink']
    else:
        if isinstance(tree_model['rlink'], dict):
            return predict(tree_model['rlink'], test_row)
        else:
            return tree_model['rlink']

def test_model(dataset, tree_model):
    predictions = list()
    for row in dataset:
        prediction = predict(tree_model, row)
        predictions.append(prediction)

    return predictions

def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions))/ len(actualValues) *100

def main():
    start_time = time.time()
    dataset = extract_full_dataset()
    print(dataset.shape)
    dataset_k_split = kfold_split(5)
    print(dataset_k_split)
    spam_accuracy = []
    count =1
    for i in dataset_k_split:
        trainingSet, testingSet = get_training_testing_split(dataset, dataset_k_split, i)
        trainingSet = trainingSet.values
        testingSet = testingSet.values
        maximum_depth = 5
        stopping_size = 10
        tree_model = build_tree(trainingSet, maximum_depth, stopping_size)
        #print("The model is", tree_model)

        predictedValues = test_model(testingSet, tree_model)

        actualValues = [row[-1] for row in testingSet]
        print("The predicted values for testing set:", predictedValues)
        print("The actual values for testing set:", actualValues)

        accuracy = evaluate_prediction_accuracy(predictedValues, actualValues)
        spam_accuracy.append(accuracy)
        print("Trial:", count)
        count+=1
        print("The accuracy here is:", accuracy)
    end_time = time.time()
    print("Individual run accuracy list:", spam_accuracy)
    print("Mean of accuracy is:", np.mean(spam_accuracy))
    print("The overall time taken is :", end_time - start_time)


if __name__ == '__main__':
    main()
