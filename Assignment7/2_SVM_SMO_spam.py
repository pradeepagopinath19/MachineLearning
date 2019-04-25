import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues) * 100


def extract_full_dataset():
    spam_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    spam_dataset = pd.read_csv(spam_dataset_url, header=None, sep=',')

    return spam_dataset


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
        if y_prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def main():
    dataset = extract_full_dataset()
    dataset = shuffle(dataset)

    # Pandas to numpy array
    dataset = dataset.values

    # print(dataset.shape)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    # {1, -1}

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1

    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainingSet = np.column_stack((X_train, y_train))
    testingSet = np.column_stack((X_test, y_test))

    alpha, bias = svm_smo(X_train, y_train)

    print(alpha, bias)
    # Prediction
    y_predications = predict_values(X_train, y_train, X_test, alpha, bias)

    # Evaluate
    accuracy = evaluate_prediction_accuracy(y_predications, y_test)
    print("Accuracy is", accuracy)


if __name__ == '__main__':
    main()
