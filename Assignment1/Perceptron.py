import numpy as np
import math
import pandas as pd
import time
import random


def fetch_dataset():
    training_url = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/perceptronData.txt"

    training_data = pd.read_csv(training_url, header=None, sep='\s+')

    #print(training_data)

    return training_data.values


def pre_process_data(dataset):
    y = [row[-1] for row in dataset]

    for i in range(len(dataset)):
        label = y[i]
        if label == -1:
            for j in range(len(dataset[0])):
                dataset[i][j] = dataset[i][j] * -1
    return dataset


def update_w_value(w, x):
    lambda_val = 0.001
    for i in range(100):
        m = []
        prediction = x.dot(w)
        for row_number in range(len(prediction)):
            if prediction[row_number] < 0:
                m.append(x[row_number,:])

        #print(m)
        for rowX in m:
            secondTerm = np.multiply(lambda_val, rowX.T)
            w = [x + y for x, y in zip(w, secondTerm)]

        print("Iteration %d, total_mistake %d" % (i + 1, len(m)))
        if len(m) == 0:
            #print(prediction)
            break
    return w


def main():
    startTime = time.time()
    dataset = fetch_dataset()

    # Converting negative vectors to lie on positive side of the plane
    clean_dataset = pre_process_data(dataset)

    # adding ones to first column
    final_dataset = np.c_[np.ones(clean_dataset.shape[0]), clean_dataset]
    # print(final_dataset)

    # initialize w values
    w = np.random.normal(size=(5,1))
    #print(w)

    # Remove labels to get x
    x = np.delete(final_dataset, final_dataset.shape[1] - 1, axis=1)

    updated_w = update_w_value(w, x)
    print("Classifier weights",w)

    # calculating w_o
    # allProducts =[]
    # for row in x:
    #     row = np.matrix(row)
    #     updated_w = np.matrix(updated_w)
    #     product = np.matmul(row, updated_w)
    #     #print(product)
    #     allProducts.append(product)
    #
    # w_o = 1- np.sum(allProducts)
    w_o = updated_w[0]
    updated_w = np.array(updated_w)
    normalized_weights = [x / -w_o for x in updated_w]
    print("Normalized weights are", normalized_weights[1:])


if __name__ == "__main__":
    main()
