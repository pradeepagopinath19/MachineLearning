import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from random import shuffle, seed


class NeuralNetwork:
    def __init__(self, x, y):
        self.hidden_size = 5
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, 3)
        self.y = y
        self.layer1 = np.random.rand(self.input.shape[1], self.hidden_size)
        self.output = np.zeros(self.y.shape)
        self.b1 = np.random.rand(1, self.hidden_size)
        self.b2 = np.random.rand(1, self.output.shape[1])
        self.loss = 0

    def feedforward(self):
        self.layer1 = self.sigmoid(np.add(np.dot(self.input, self.weights1), self.b1))
        self.output = self.sigmoid(np.add(np.dot(self.layer1, self.weights2), self.b2))

    def calculate_loss(self):
        self.loss = self.calculate_mean_square_loss(self.output, self.y)

    def calculate_mean_square_loss(self, estimation, true_values):
        errorValues = np.array(estimation) - np.array(true_values)
        sum = np.sum(np.square(errorValues))
        return np.mean(sum)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


def fetchDataset():
    training_url = "train_wine.csv"
    testing_url = "test_wine.csv"
    training_data = pd.read_csv(training_url, header=None, sep=',')
    testing_data = pd.read_csv(testing_url, header=None, sep=',')
    return training_data, testing_data


def calculate_accuracy(pred, true):
    correct_predictions = [i for i, j in zip(pred, true) if i == j]
    return len(correct_predictions) / len(true) * 100

def one_hot(x):
    if x == 1:
        return [0,0,1]
    elif x ==2:
        return [0,1,0]
    else:
        return [1,0,0]

def run_neural_networks():
    epochs = 1000
    learning_rate = 0.1
    training, testing = fetchDataset()

    X_train = training.iloc[:, 1:training.shape[1]].values
    X_test = testing.iloc[:, 1:testing.shape[1]].values
    Y_train = training.iloc[:, 0].values
    Y_test = testing.iloc[:, 0].values

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    Y_train = list(map(lambda x: one_hot(x), Y_train))
    Y_test = list(map(lambda x:one_hot(x), Y_test))
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)
    #print(Y_test,Y_train)
    nn = NeuralNetwork(X_train, Y_train)
    for _ in range(epochs):
        # Forward propagation
        nn.feedforward()
        nn.calculate_loss()
        print(nn.loss)

        # Back propagation

        error = nn.y - nn.output
        slope_output = nn.derivative_sigmoid(nn.output)
        d_output = error * slope_output
        error_hidden = d_output.dot(nn.weights2.T)

        slope_hidden = nn.derivative_sigmoid(nn.layer1)
        d_hidden = error_hidden * slope_hidden

        # updating weights

        nn.weights2 += nn.layer1.T.dot(d_output) * learning_rate
        nn.weights1 += nn.input.T.dot(d_hidden) * learning_rate

        # updating biases

        nn.b1 += np.sum(a=d_hidden, axis=0, keepdims=True) * learning_rate
        nn.b2 += np.sum(a=d_output, axis=0, keepdims=True) * learning_rate

    # print(nn.output)
    # print(nn.y)
    # print(nn.layer1)

    prediction = np.argmax(nn.output, axis=1)
    true_value = np.argmax(nn.y, axis=1)

    #print(prediction, true_value)
    accuracy = calculate_accuracy(prediction, true_value)
    print('Training accuracy is', accuracy)

    testing_nn = NeuralNetwork(X_test, Y_test)
    testing_nn.weights1 = nn.weights1
    testing_nn.weights2 = nn.weights2
    testing_nn.b1= nn.b1
    testing_nn.b2 = nn.b2
    testing_nn.feedforward()
    #print(testing_nn.output)
    prediction = np.argmax(testing_nn.output, axis=1)
    true_value = np.argmax(testing_nn.y, axis=1)

    # print(prediction, true_value)
    accuracy = calculate_accuracy(prediction, true_value)
    print('Testing accuracy is', accuracy)

if __name__ == '__main__':
    run_neural_networks()
