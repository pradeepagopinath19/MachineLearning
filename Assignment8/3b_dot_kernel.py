import numpy as np
import math
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split


def fetch_dataset():
    dataset_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/TwoSpirals/twoSpirals.txt"
    dataset_data = pd.read_csv(dataset_url, header=None, sep='\s+')

    return dataset_data.values


def compute_mx(m, X):
    print(m.shape)
    return m * np.inner(X, X)


def main():
    dataset = fetch_dataset()
    # print(dataset.shape)

    # No pre processing for dual perceptron
    m = np.zeros((len(dataset), 1))

    # print(m.shape)
    # print(m)
    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    for iter in range(1000):
        # prediction
        m_x = compute_mx(m, X)
        m_x = np.sum(m_x, axis=1)
        prediction = y * m_x
        prediction = prediction.flatten()
        indices = [i for i in range(len(prediction)) if prediction[i] <= 0]
        # print(indices)
        print(len(indices))
        if len(indices) == 0:
            print("Success after", iter + 1, "iterations")
            break

        for index in indices:
            m[index] += y[index]


if __name__ == '__main__':
    main()
