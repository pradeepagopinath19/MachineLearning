import numpy as np
import math
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.metrics.pairwise import rbf_kernel


def fetch_dataset():
    dataset_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/TwoSpirals/twoSpirals.txt"
    dataset_data = pd.read_csv(dataset_url, header=None, sep='\s+')

    return dataset_data.values


def calculate_gaussian_kernel_distance(xi, x):
    sigma = 1.0
    return np.exp(-(spatial.distance.euclidean(xi, x) ** 2) / (2 * (sigma ** 2)))


def compute_mx(m, X):
    print(m.shape)
    return m * rbf_kernel(X,X)

def main():
    dataset = fetch_dataset()
    print(dataset.shape)

    # No pre processing for dual perceptron
    m = np.zeros((len(dataset), 1))

    # print(m.shape)
    # print(m)
    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    for iter in range(2500):
        # prediction

        m_x = compute_mx(m, X)
        #print(m_x.shape)
        #exit()
        m_x = np.sum(m_x, axis=1)
        prediction = y * m_x
        print(prediction.shape)
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
