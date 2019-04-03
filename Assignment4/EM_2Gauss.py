import numpy as np
import pandas as pd
import random
import math
import ConfusionMatrix
from sklearn.utils import shuffle


def extract_full_dataset():
    url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/3_generative_models/HW3/2gaussian.txt"
    dataset = pd.read_csv(url, header=None, sep='\s+')

    return dataset.values


def main():
    # Extracting data

    X = extract_full_dataset()
    #print(X.shape)

    # Initialization

    num_of_iterations = 500
    number_of_features = 2
    number_of_classification = 2


    # To check convergence
    llh=[]


    z_im = np.zeros((X.shape[0], number_of_classification))

    # n * k (n : Number of features, k: number_of_classification)
    mu = np.random.randint(min(X[:, 1]), max(X[:, 1]), size=(number_of_features, number_of_features))
    print(mu.shape)

    # n * k * k
    cov = np.zeros((number_of_features, number_of_classification, number_of_classification))
    #print(cov)

    # Filling the diagonals

    for row in range(len(cov)):
        np.fill_diagonal(cov[row], 4)

    #print(cov)


    # Algorithm implementation
    # while True:
    #     for i in range(number_of_features):
    #         for m in range(number_of_classification):
    #             break


        # if has_it_converged():
        #     break



if __name__ == '__main__':
    main()
