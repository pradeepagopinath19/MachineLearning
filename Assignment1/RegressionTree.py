import numpy as np
import math
import pandas as pd
import time




def split_data(dataset, featureColumn, thresholdRow):

    thresholdValue = dataset[thresholdRow][featureColumn]
    #print(thresholdValue)
    #print(featureColumnValues)

    left, right = [],[]
    for row in dataset:
        if row[featureColumn] < thresholdValue:
            left.append(row)
        else:
            right.append(row)
    return left, right

def calculate_mse(leftValues, rightValues, label):

    leftLabelValues = [row[-1] for row in leftValues]
    rightLabelValues = [row[-1] for row in rightValues]


    if len(leftLabelValues) != 0:
        varianceLeft = np.var(leftLabelValues)
    else:
        varianceLeft = 0

    if len(rightLabelValues) !=0:
        varianceRight = np.var(rightLabelValues)
    else:
       varianceRight = 0

    return (varianceLeft * len(leftLabelValues)/len(label)) + (varianceRight * len(rightLabelValues) / len(label))

def find_best_split(dataset):
    labelValues = [row[-1] for row in dataset]

    minimumError = math.inf
    minimumFeature = 0
    minimumThreshold = 0
    left, right = [], []

    for j in range(len(dataset[0])-1):
        for i in range(len(dataset)):
            left, right = split_data(dataset, j, i)
            calculatedError = calculate_mse(left, right, labelValues)
            if calculatedError < minimumError:
                minimumError = calculatedError
                minimumFeature = j
                minimumThreshold = dataset[i][j]
    print("Feature: %d minimumThreshold: %f Error: %f" % (minimumFeature+1, minimumThreshold, minimumError))
    return {'left': left,'right': right, 'feature': minimumFeature, 'threshold': minimumThreshold, 'error': minimumError}


def leaf_node(data):
    featureValues = [row[-1] for row in data]

    mean =0
    if len(featureValues)>0:
        mean = np.mean(featureValues)
    return mean


def split_tree(node, max_depth, stopping_size, depth):
    #print(node)

    #stopping criteria

    if depth == max_depth:
        node['llink']=leaf_node(node['left'])
        node['rlink']= leaf_node(node['right'])
        return

    if not node['left'] or not node['right']:
        node['llink'] = node['rlink'] = leaf_node(node['left']+node['right'])
        return

    # recursion steps
    if len(node['left']) <= stopping_size:
        node['llink'] = leaf_node(node['left'])
    else:
        node['llink']= find_best_split(node['left'])
        split_tree(node['llink'], max_depth, stopping_size, depth+1)

    if len(node['right']) <= stopping_size:
        node['rlink'] = leaf_node(node['right'])
    else:
        node['rlink'] = find_best_split(node['right'])
        split_tree(node['rlink'], max_depth, stopping_size, depth+1)


def build_tree(dataset, max_depth, stopping_size):
    root_node = find_best_split(dataset)
    print(root_node)
    split_tree(root_node, max_depth, stopping_size, 1)
    return root_node


def fetch_dataset():

    training_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt"
    testing_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt"
    training_data = pd.read_csv(training_url, header=None, sep='\s+')
    testing_data = pd.read_csv(testing_url, header=None, sep='\s+')

    #print(training_data)
    #print(testing_data)
    return training_data.values, testing_data.values




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
    return (predictions)


def evaluate_prediction(estimation, dataset):
    trueValues =[row[-1] for row in dataset]
    errorValues = np.array(estimation)-np.array(trueValues)
    return (np.sum(np.square(errorValues))/len(dataset))

def main():

    startTime = time.time()
    training_dataset, testing_dataset = fetch_dataset()
    #print(training_dataset)
    #print(testing_dataset)

    maximum_depth = 5
    stopping_size = 10

    tree_model = build_tree(training_dataset, maximum_depth, stopping_size)
    print("The model is", tree_model)

    predictedValues = test_model(testing_dataset, tree_model)
    print("The predicted values are",predictedValues)

    evaluateModel_mse = evaluate_prediction(predictedValues, testing_dataset)
    print("The calculated MSE is", evaluateModel_mse)

    endTime = time.time()
    print("Operations completed in", endTime-startTime)


if __name__ == "__main__":
    main()