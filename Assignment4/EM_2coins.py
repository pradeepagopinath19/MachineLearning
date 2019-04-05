import numpy as np
import pandas as pd
import random
import math
import ConfusionMatrix
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal
import matplotlib as plt


def create_coin_flips(theta, pi, k, n):
    flip_seq = []
    choice_coin = np.random.binomial(1, pi, n)
    coin_prob = 0
    for c in choice_coin:
        each_trial = []
        if c == 0:
            coin_prob = theta[0]
        else:
            coin_prob = theta[1]
        each_trial = np.random.binomial(1, coin_prob, k)
        flip_seq.append(each_trial)
    print(flip_seq)
    return flip_seq


def main():
    p = 0.75
    r = 0.4
    pi = 0.8

    theta = [p, r]
    # print(theta)

    # Given
    max_iteration = 1000
    n = 1000
    k = 10

    # Create coin_flips

    flip_sequence = create_coin_flips(theta, pi, k, n)

    for i in range(max_iteration):



if __name__ == '__main__':
    main()
