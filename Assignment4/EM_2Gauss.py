import numpy as np
import pandas as pd
import random
import math
import ConfusionMatrix
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal
import matplotlib as plt


def extract_full_dataset():
    url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/3_generative_models/HW3/2gaussian.txt"
    dataset = pd.read_csv(url, header=None, sep='\s+')

    return dataset.values


def main():
    # Extracting data

    X = extract_full_dataset()
    X = shuffle(X, random_state=0)
    # print(X.shape)

    # Initialization

    num_of_iterations = 1500
    number_of_features = 2
    number_of_classification = 2

    # To check convergence
    llh = []

    # n * k (n : Number of features, k: number_of_classification)
    mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(number_of_features, number_of_features))
    # print(mu.shape)

    # n * k * k
    cov = np.zeros((number_of_features, number_of_classification, number_of_classification))
    # print(cov)

    # [0.5,0.5]
    pi = np.ones(number_of_classification) / number_of_classification

    # Filling the diagonals

    for row in range(len(cov)):
        np.fill_diagonal(cov[row], 5)

    # print(cov)

    for i in range(num_of_iterations):
        # E step

        z_im = np.zeros((len(X), len(cov)))

        # Getting the z_im value that sum to one.

        for mean, c, w, column_z in zip(mu, cov, pi, range(len(z_im[0]))):
            print(mean.shape)
            norm = multivariate_normal(mean=mean, cov=c)
            prob_df = norm.pdf(x=X)
            numerator = w * prob_df

            # Sum of pdf of point belonging to
            denominator = 0

            for mean_k, pi_k, cov_k in zip(mu, pi, cov):
                denominator += multivariate_normal(mean_k, cov_k).pdf(X) * pi_k
            z_im[:, column_z] = numerator / denominator
        # print(z_im)

        # M step

        # Resetting the values of mu, cov, pi
        mu = []
        cov = []
        pi = []

        for m in range(len(z_im[0])):
            sum_zim = np.sum(z_im[:, m], axis=0)
            mu_m = np.sum(X * z_im[:, m].reshape(len(X), 1), axis=0)

            # mean
            mu.append(mu_m / sum_zim)

            # Cov
            cov.append(np.dot((np.array(z_im[:, m]).reshape(len(X), 1) * (X - mu_m)).T, (X - mu_m)) / sum_zim)

            # pi_m

            pi.append(sum_zim / np.sum(z_im))

        llh.append(np.log(np.sum(
            [k * multivariate_normal(mu[i], cov[j]).pdf(X) for k, i, j in zip(pi, range(len(mu)), range(len(cov)))])))
    print(llh)
    print(mu)
    print(cov)
    count_n1 = 0
    for first, second in z_im:
        if first>second:
            count_n1+=1
    print("n1 = ", count_n1 ,"points")
    print("n2 = ", 6000 - count_n1, "points")


if __name__ == '__main__':
    main()
