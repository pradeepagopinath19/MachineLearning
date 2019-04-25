import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle

def extract_haar_features():
    training_dataset = pd.read_csv("Haar_feature_full_training.csv", header=None, sep=',')
    testing_dataset = pd.read_csv("Haar_feature_testing.csv", header=None, sep=',')
    return training_dataset.values, testing_dataset.values


def compute_one_vs_rest_training(trainingSet, class_label):
    local_training_dataset = np.copy(trainingSet)

    for i in range(len(local_training_dataset)):
        if local_training_dataset[i][-1] != class_label:
            local_training_dataset[i][-1] = -1
    return local_training_dataset


def select_random_j_value(i, max_val):
    j = int(np.random.randint(0, max_val))
    while j == i:
        j = int(np.random.uniform(0, max_val))

    return j


def calculate_eta(X, i, j):
    return 2.0 * X[i, :] * X[j, :].T - (X[i, :] * X[j, :].T) - X[j, :] * X[j, :].T
    # return (2 * np.dot(x_i, x_j)) - np.dot(x_i, x_i) - np.dot(x_j, x_j)


def calculate_alpha_j(alpha_j_old, y_j, e_i, e_j, eta, h, l):
    alpha_j = alpha_j_old - (((y_j) * (e_i - e_j)) / eta)

    if alpha_j > h:
        return h
    elif l > alpha_j:
        return l
    else:
        return alpha_j


def calculate_b1_b2(b, y_i, y_j, alpha_i, alpha_i_old, alpha_j, alpha_j_old, x_i, x_j, e_i, e_j):
    b1 = b - e_i - y_i * (alpha_i - alpha_i_old) * x_i * x_i.T - y_j * (alpha_j - alpha_j_old) * x_i * x_j.T
    b2 = b - e_j - y_i * (alpha_i - alpha_i_old) * x_i * x_j.T - y_j * (alpha_j - alpha_j_old) * x_j * x_j.T

    return b1, b2


def compute_final_b(alpha_i, alpha_j, c, b1, b2):
    if alpha_i > 0 and alpha_i < c:
        return b1
    elif alpha_j > 0 and alpha_j < c:
        return b2
    else:
        return (b1 + b2) / 2.0


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def svm_smo(X, y):
    y = np.mat(y).transpose()
    X = np.mat(X)
    # Initialization
    c = 0.001
    tolerance = 0.01
    epsilon = 0.001
    number_of_iterations = 100
    alpha = np.mat(np.zeros((X.shape[0], 1)))
    b = 0
    # b = np.mat([[0]])
    # print(alpha, b)

    m, n = X.shape
    iter = 0
    while iter < number_of_iterations:
        alpha_changed = 0
        for i in range(m):
            # print("b values is", b)
            fxi = float(np.multiply(alpha, y).T * (X * X[i, :].T)) + b
            e_i = fxi - float(y[i])
            # e_i = np.multiply(y, alpha).T * X * X.T + b - y[i]
            # print(e_i)
            if ((y[i] * e_i < -tolerance) and (alpha[i] < c)) or ((y[i] * e_i > tolerance) and (alpha[i] > 0)):
                j = select_random_j_value(i, m)
                # print(i, j)

                fxj = float(np.multiply(alpha, y).T * (X * X[j, :].T)) + b
                e_j = fxj - float(y[j])

                # saving the old values - deep copy
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                if y[i] != y[j]:
                    l = max(0, alpha[j] - alpha[i])
                    h = min(c, c + alpha[j] - alpha[i])

                else:
                    l = max(0, alpha[j] + alpha[i] - c)
                    h = min(c, alpha[j] + alpha[i])

                if l == h:
                    continue

                eta = calculate_eta(X, i, j)
                if eta >= 0:
                    continue
                alpha[j] = calculate_alpha_j(alpha[j], y[j], e_i, e_j, eta, h, l)

                if abs(alpha[j] - alpha_j_old) < epsilon:
                    continue

                alpha[i] += y[j] * y[i] * (alpha_j_old - alpha[j])

                b1, b2 = calculate_b1_b2(b, y[i], y[j], alpha[i], alpha_i_old, alpha[j], alpha_j_old, X[i, :], X[j, :],
                                         e_i, e_j)
                b = compute_final_b(alpha[i], alpha[j], c, b1, b2)
                alpha_changed += 1
        if alpha_changed == 0:
            iter += 1
        else:
            iter = 0

        return alpha, b


def predict_values(X_train, y_train, X_test, alpha, bias):
    predictions = []
    y = np.mat(y_train).transpose()
    X = np.mat(X_train)
    X_test = np.mat(X_test)
    for i in range(len(X_test)):
        y_prediction = float(np.multiply(alpha, y).T * (X * X_test[i, :].T)) + bias
        predictions.append(y_prediction)
        # Returning the raw values. Not making the predictions just yet.
        # if y_prediction >= 0:
        #     predictions.append(1)
        # else:
        #     predictions.append(-1)
    return np.array(predictions).reshape((len(X_test), 1))


def main():
    trainingSet, testingSet = extract_haar_features()
    trainingSet = shuffle(trainingSet)
    # Pre computed values from Assignment 5
    # Hard coded for efficiency reasons

    label_count = {5: 5421, 0: 5923, 4: 5842, 1: 6742, 9: 5949, 2: 5958, 3: 6131, 6: 5918, 7: 6265, 8: 5851}

    num_label_local = {}

    new_trainingset = []
    for row in trainingSet:
        label = row[-1]
        if label in num_label_local:
            if num_label_local[label] >= label_count[label] * 0.3:
                continue
            num_label_local[label] += 1
            new_trainingset.append(row)
        else:
            num_label_local[label] = 1
            new_trainingset.append(row)

    trainingSet = np.array(new_trainingset)

    print(trainingSet.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(trainingSet[:, 0:-1])
    trainingSet[:, 0:-1] = scaler.transform(trainingSet[:, 0:-1])

    scaler = preprocessing.StandardScaler()
    scaler.fit(testingSet[:, 0:-1])
    testingSet[:, 0:-1] = scaler.transform(testingSet[:, 0:-1])

    # one vs rest implementation
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_training_data = {}

    alpha_classlabel = {}
    bias_classlabel = {}

    for class_label in class_labels:
        class_training_data[class_label] = compute_one_vs_rest_training(trainingSet, class_label)

        data = class_training_data[class_label]
        # calling SMO on the 10 binary classifier
        alpha_classlabel[class_label], bias_classlabel[class_label] = svm_smo(data[:, 0:-1], data[:, -1])
        print(class_label, "done")

    # # Writing it into a file for making the predictions later on
    # f = open("alpha.txt", "w")
    # f.write(str(alpha_classlabel))
    # f.close()
    #
    # f = open("bias.txt", "w")
    # f.write(str(bias_classlabel))
    # f.close()
    #
    # print("Done")
    #
    #
    #
    # # Reading the dictionary
    # alpha_classlabel = eval(open('alpha.txt', 'r').read())
    # bias_classlabel = eval(open('bias.txt', 'r').read())
    #
    X_test = testingSet[:, 0:-1]
    y_test = testingSet[:, -1]

    predictions = {}
    for class_label in class_labels:
        data = class_training_data[class_label]
        predictions[class_label] = predict_values(data[:, 0:-1], data[:, -1], X_test, alpha_classlabel[class_label],
                                                  bias_classlabel[class_label])

    # predicting best values from all the models
    y_predications = {}
    for class_label in class_labels:
        for i, val in enumerate(predictions[class_label]):
            if i in y_predications:
                if val > y_predications[i][0]:
                    y_predications[i] = (val, class_label)
            else:
                y_predications[i] = (val, class_label)

    y_labels = [0] * len(y_test)

    for i in range(len(y_test)):
        val, class_label = y_predications[i]
        y_labels[i] = class_label

    print(y_labels)
    print(y_test)
    print(len(y_labels))
    print(len(y_test))
    # Evaluate
    accuracy = evaluate_prediction_accuracy(y_labels, y_test)
    print("Accuracy is", accuracy)


if __name__ == '__main__':
    main()
