import numpy as np
import scipy
import pandas as pd
import random
import math
import ConfusionMatrix
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal
import matplotlib as plt


def create_coin_flips(theta, pi, k, n):
    flip_seq = []
    choice_coin = np.random.binomial(1, pi[1], n)
    #print(choice_coin)
    coin_prob = 0
    for c in choice_coin:
        each_trial = []
        if c == 0:
            coin_prob = theta[0]
        else:
            coin_prob = theta[1]
        each_trial = np.random.binomial(1, coin_prob, k)
        flip_seq.append(each_trial)
    # print(flip_seq)
    return flip_seq


def calculate_stats(flip):
    heads, tails = 0, 0
    for trial in flip:
        if trial == 1:
            heads += 1
    return heads


def main():
    p = 0.75
    r = 0.4
    pi = [0.8, 0.2]
    number_of_coins = 2

    theta = [p, r]
    # print(theta)

    # Given
    max_iteration = 500
    n = 1000
    k = 10

    # Create coin_flips

    X = create_coin_flips(theta, pi, k, n)

    for _ in range(max_iteration):
        z_nk = np.zeros((len(X), number_of_coins))
        no_heads = []
        no_heads_n = 0
        # E step
        for i in range(len(X)):
            no_heads_n = calculate_stats(X[i])
            no_heads.append(no_heads_n)

            # print(no_heads)
            binomial_coin1 = scipy.stats.binom.pmf(no_heads_n, k, theta[0]) * pi[0]
            binomial_coin2 = scipy.stats.binom.pmf(no_heads_n, k, theta[1]) * pi[1]

            sum_binomial = binomial_coin1 + binomial_coin2

            normalized_a = binomial_coin1 / sum_binomial
            normalized_b = binomial_coin2 / sum_binomial
            # print(normalized_a, normalized_b)

            z_nk[i][0] = normalized_a
            z_nk[i][1] = normalized_b

        # M steps

        pi[0] = np.sum(z_nk[:, 0]) / n
        pi[1] = np.sum(z_nk[:, 1]) / n

        numerator_0 = 0
        denominator_0 = 0

        numerator_1 = 0
        denominator_1 = 0
        for z, y_n in zip(z_nk, no_heads):
            numerator_0 += z[0] * y_n
            denominator_0 += z[0] * k

            numerator_1 += z[1] * y_n
            denominator_1 += z[1] * k
        theta[0]= numerator_0/denominator_0
        theta[1]= numerator_1/denominator_1
    print("Pi value is", pi)
    print("Theta value is", theta)


if __name__ == '__main__':
    main()
