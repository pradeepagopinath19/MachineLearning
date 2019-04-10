import numpy as np
import pandas as pd
import random
from DecisionStump import DecisionStump
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from random import randrange
import matplotlib.pyplot as plt

def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset

def main():
    dataset = extract_full_dataset()
    dataset = dataset.values
    X = dataset[:, 0:57]
    y = dataset[:, -1]

    spam_dataset = shuffle(dataset)


if __name__ == '__main__':
    main()
