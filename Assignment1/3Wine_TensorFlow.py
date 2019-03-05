 import tensorflow as tf
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def extract_dataset():
    training_url = "train_wine.csv"
    testing_url = "test_wine.csv"
    training_data = pd.read_csv(training_url, header=None, sep=',', )
    testing_data = pd.read_csv(testing_url, header=None, sep=',')
    return training_data, testing_data


def main():
    np.random.seed(3)
    classifications = 3


    training, testing = extract_dataset()
    # print(training, testing)

    X_train = training.iloc[:,1:training.shape[1]].values
    X_test = testing.iloc[:, 1:testing.shape[1]].values
    Y_train = training.iloc[:,0].values
    Y_test = testing.iloc[:,0].values

    Y_train = Y_train.reshape(-1,1)
    Y_test = Y_test.reshape(-1,1)

    Y_train = np.array([labelMaker(i[0]) for i in Y_train])
    Y_test = np.array([labelMaker(i[0]) for i in Y_test])

    #print(Y_train, Y_test)

    #model

    model = Sequential()
    model.add(Dense(10, input_dim=13, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(classifications, activation='softmax'))

    # compile and fit model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=15, epochs=2500, validation_data=(X_test, Y_test))


def labelMaker(val):
    if val == 1:
        return [1, 0, 0]
    elif val == 2:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

if __name__ == '__main__':
    main()