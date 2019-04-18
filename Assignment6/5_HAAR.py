import pandas as pd
import numpy as np
import numbers
import re
import random
import math
import collections
from sklearn.utils import shuffle
from DecisionStump import DecisionStump
import os
from Random_rectangle import Rectangle
from random import randrange
from sklearn.metrics import roc_auc_score
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros


def extract_haar_features():
    training_dataset = pd.read_csv("Haar_feature_training.csv", header=None, sep=',')
    # training_dataset = pd.read_csv("Haar_feature_full_training.csv", header=None, sep=',')
    testing_dataset = pd.read_csv("Haar_feature_testing.csv", header=None, sep=',')
    return training_dataset.values, testing_dataset.values


def load_mnist(dataset="training", digits=np.arange(10), path='.'):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    # ind = [ k for k in range(size) ]
    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels


def get_origin_rectangles_testing(images, labels, n):
    black_pixel_images = []
    label_images = []
    for image, label in zip(images, labels):
        black_pixel_rect = np.zeros((n, n))
        for i in range(1, n):
            for j in range(1, n):
                if image[i, j] > 0:
                    black_pixel_rect[i, j] = black_pixel_rect[i, j - 1] + black_pixel_rect[i - 1, j] - black_pixel_rect[
                        i - 1, j - 1] + 1
                else:
                    black_pixel_rect[i, j] = black_pixel_rect[i, j - 1] + black_pixel_rect[i - 1, j] - black_pixel_rect[
                        i - 1, j - 1]
        black_pixel_images.append(black_pixel_rect)
        label_images.append(int(label))
    return black_pixel_images, label_images


def get_origin_rectangles_training(images, labels, label_count, n):
    black_pixel_images = []
    # Sampling 20% of each label
    local_label_count = {}
    label_images = []
    for image, label in zip(images, labels):
        if int(label) in local_label_count:
            if local_label_count[int(label)] >= int(0.2 * label_count[int(label)]):
                continue
            else:
                local_label_count[int(label)] += 1
        else:
            local_label_count[int(label)] = 1
        black_pixel_rect = np.zeros((n, n))
        for i in range(1, n):
            for j in range(1, n):
                if image[i, j] > 0:
                    black_pixel_rect[i, j] = black_pixel_rect[i, j - 1] + black_pixel_rect[i - 1, j] - black_pixel_rect[
                        i - 1, j - 1] + 1
                else:
                    black_pixel_rect[i, j] = black_pixel_rect[i, j - 1] + black_pixel_rect[i - 1, j] - black_pixel_rect[
                        i - 1, j - 1]
        black_pixel_images.append(black_pixel_rect)
        label_images.append(int(label))
    return black_pixel_images, label_images


def generate_random_rectangles(n, count):
    random_rectangles = []

    while len(random_rectangles) < count:
        top_left = [np.random.randint(n, size=1) for _ in range(2)]
        bottom_right = [np.random.randint(n, size=1) for _ in range(2)]

        rect = Rectangle(top_left, bottom_right)
        if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
            # Area should be between 130-170 area
            if rect.compute_area() >= 130 and rect.compute_area() <= 170:
                random_rectangles.append(rect)
    return random_rectangles


def calculate_black_in_rectangle(black_rect_count, rectangle):
    black_sum = black_rect_count[rectangle.bottom_right[0], rectangle.bottom_right[1]] - black_rect_count[
        rectangle.top_right[0], rectangle.top_right[1]] - black_rect_count[
                    rectangle.bottom_left[0], rectangle.bottom_left[1]] + black_rect_count[
                    rectangle.top_left[0], rectangle.top_left[1]]

    return black_sum


def fetch_features(origin_rectangles, random_rectangles):
    features = []
    for black_rect in origin_rectangles:
        # print(black_rect)
        image_features = []
        for r in random_rectangles:
            top_vert_rect = Rectangle(r.top_left, [r.bottom_right[0], r.top_right[1] + (r.height // 2)])
            bottom_vert_rect = Rectangle([r.top_left[0], r.top_left[1] + (r.height // 2)], r.bottom_right)
            left_hor_rect = Rectangle(r.top_left, [r.top_left[0] + (r.width // 2), r.bottom_left[1]])
            right_hor_rect = Rectangle([r.top_left[0] + (r.width // 2), r.top_left[1]], r.bottom_right)

            black_top_vertical = calculate_black_in_rectangle(black_rect, top_vert_rect)
            black_bottom_vertical = calculate_black_in_rectangle(black_rect, bottom_vert_rect)
            black_left_horizontal = calculate_black_in_rectangle(black_rect, left_hor_rect)
            black_right_horizontal = calculate_black_in_rectangle(black_rect, right_hor_rect)

            image_features.append(black_top_vertical - black_bottom_vertical)
            image_features.append(black_left_horizontal - black_right_horizontal)
        features.append(image_features)
    return features


def random_ECOC_generator(k=10, n=50):
    labels = []
    for _ in range(k):
        labels.append(np.random.randint(0, 2, n, dtype=int))

    return labels


def fetch_label_code():
    # random_label_values = random_ECOC_generator(k=10, n=50)
    # random_label_values = pd.DataFrame(random_label_values, columns=None)
    # random_label_values.to_csv('ECOC_Label_code.txt', header=None, index=False)
    label_code = pd.read_csv("ECOC_Label_code.txt", header=None, sep=',').values

    return label_code


def minimum_dist(prediction, label_code):
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    for label in label_code:
        min_val = 51
        minimum_label_val = []
        diff = sum(1 for a, b in zip(label, prediction) if a != b)

        if diff < min_val:
            min_val = diff
            minimum_label_val = label

        if diff == 0:
            break

    for i, value in enumerate(label_code):
        if compare(value, minimum_label_val):
            return i


def fetch_minimum_distance(test_prediction, label_code):
    final_prediction = []

    for prediction in test_prediction:
        best_label = minimum_dist(prediction, label_code)
        final_prediction.append(best_label)
        # print(best_label)
    return final_prediction


def evaluate_prediction_accuracy(predictedValues, actualValues):
    correct_predictions = [i for i, j in zip(predictedValues, actualValues) if i == j]

    return float(len(correct_predictions)) / len(actualValues)


def predict(classifiers, X):
    y_pred = np.zeros((len(X), 1))

    for c in classifiers:
        non_spam_idx = (c.polarity * X[:, c.feature] < c.polarity * c.threshold)
        # print(non_spam_idx)

        predictions = np.ones((len(X), 1))
        predictions[non_spam_idx] = -1
        y_pred += c.alpha * predictions

    return np.sign(y_pred).flatten()


def adaboost_algo(training_x, training_y, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(training_x)) / len(training_x)

    dec_classifiers = []

    for iter_number in range(max_iter):

        classifier = DecisionStump()
        min_weighted_error = math.inf

        # Best decision stump
        for j in range(len(training_x[0])):

            f_values = training_x[:, j]
            unique_feature = set(f_values)

            for threshold in unique_feature:
                stump_prediction = np.ones((np.shape(training_y)))
                stump_prediction[f_values < threshold] = -1

                weighted_error = np.sum(w[training_y != stump_prediction])

                if weighted_error > 0.5:
                    p = -1
                    weighted_error = 1 - weighted_error
                else:
                    p = 1

                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error

                    classifier.threshold = threshold
                    classifier.feature = j
                    classifier.polarity = p
        classifier.alpha = 0.5 * math.log((1.0 - min_weighted_error) / (min_weighted_error + 1e-10))

        predictions = np.ones(training_y.shape)
        negative_idx = (
                classifier.polarity * training_x[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w
        # print(w.shape, y_train.shape, predictions.shape)
        # print(type(w), type(y_train), type(predictions))
        w *= np.exp(-classifier.alpha * training_y * predictions)

        w /= np.sum(w)

        dec_classifiers.append(classifier)

        #Printing and verification after each step

        prediction_y_train = predict(dec_classifiers, training_x)
        prediction_y_test = predict(dec_classifiers, testing_x)

        training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
        testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)

        auc_val = roc_auc_score(testing_y, prediction_y_test)

        print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
              "Weighted error", min_weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
              1 - testing_accuracy,
              "AUC", auc_val)

    return dec_classifiers


def adaboost_algo_interval(training_x, training_y, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(training_x)) / len(training_x)

    dec_classifiers = []

    for iter_number in range(max_iter):

        classifier = DecisionStump()
        min_weighted_error = math.inf

        # Best decision stump
        for j in range(len(training_x[0])):

            f_values = training_x[:, j]
            unique_feature = set(f_values)
            unique_feature_linespace = np.linspace(min(unique_feature), max(unique_feature), num=6)

            for threshold in unique_feature_linespace:
                stump_prediction = np.ones((np.shape(training_y)))
                stump_prediction[f_values < threshold] = -1

                weighted_error = np.sum(w[training_y != stump_prediction])

                if weighted_error > 0.5:
                    p = -1
                    weighted_error = 1 - weighted_error
                else:
                    p = 1

                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error

                    classifier.threshold = threshold
                    classifier.feature = j
                    classifier.polarity = p
        classifier.alpha = 0.5 * math.log((1.0 - min_weighted_error) / (min_weighted_error + 1e-10))

        predictions = np.ones(training_y.shape)
        negative_idx = (
                classifier.polarity * training_x[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w
        # print(w.shape, y_train.shape, predictions.shape)
        # print(type(w), type(y_train), type(predictions))
        w *= np.exp(-classifier.alpha * training_y * predictions)

        w /= np.sum(w)

        dec_classifiers.append(classifier)

        # Printing and verification after each step

        # prediction_y_train = predict(dec_classifiers, training_x)
        # prediction_y_test = predict(dec_classifiers, testing_x)
        #
        # training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)
        #
        # auc_val = roc_auc_score(testing_y, prediction_y_test)
        #
        # print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
        #       "Weighted error", min_weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
        #       1 - testing_accuracy,
        #       "AUC", auc_val)

    return dec_classifiers


def adaboost_algo_random(x_train, y_train, testing_x, testing_y, max_iter):
    # Initialize weights to 1/n initially
    w = np.ones(len(x_train)) / len(x_train)

    dec_classifiers = []
    weighted_error = math.inf

    for iter_number in range(max_iter):
        classifier = DecisionStump()

        feature = randrange(0, len(x_train[0]))
        f_values = x_train[:, feature]
        unique_feature = set(f_values)
        unique_feature = list(unique_feature)
        random_index = randrange(len(unique_feature))
        threshold_val = unique_feature[random_index]
        stump_prediction = np.ones((np.shape(y_train)))
        stump_prediction[f_values < threshold_val] = -1
        weighted_error = np.sum(w[y_train != stump_prediction])

        if weighted_error > 0.5:
            p = -1
            weighted_error = 1 - weighted_error
        else:
            p = 1
        classifier.threshold = threshold_val
        classifier.feature = feature
        classifier.polarity = p
        classifier.alpha = 0.5 * math.log((1.0 - weighted_error) / (weighted_error + 1e-10))

        predictions = np.ones(y_train.shape)
        negative_idx = (
                classifier.polarity * x_train[:, classifier.feature] < classifier.polarity * classifier.threshold)
        predictions[negative_idx] = -1

        # Updating w

        w *= np.exp(-classifier.alpha * y_train * predictions)

        w /= np.sum(w)

        dec_classifiers.append(classifier)

        # Printing and verification after each step

        # prediction_y_train = predict(dec_classifiers, x_train)
        # prediction_y_test = predict(dec_classifiers, testing_x)

        # training_accuracy = evaluate_prediction_accuracy(y_train, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)

        # auc_val = roc_auc_score(testing_y, prediction_y_test)

        # print("Round number", iter_number, "Feature:", classifier.feature, "Threshold:", classifier.threshold,
        #       "Weighted error", weighted_error, "Training_error", 1 - training_accuracy, "Testing_error",
        #       1 - testing_accuracy,
        #       "AUC", auc_val)

    return dec_classifiers


def main():
    # count_random_rectangle = 100
    # training_images, training_labels = load_mnist()
    # testing_images, testing_labels = load_mnist(dataset="testing")
    #
    # # Calulating the class priors
    # # for i in training_labels:
    # #     val = list(i)[0]
    # #     if val in label_count:
    # #         label_count[val]+=1
    # #     else:
    # #         label_count[val] = 1
    #
    # # For fast execution, using the values directly.
    # label_count = {5: 5421, 0: 5923, 4: 5842, 1: 6742, 9: 5949, 2: 5958, 3: 6131, 6: 5918, 7: 6265, 8: 5851}
    #
    # n = len(training_images[0])
    #
    # # Rectangle from the origin
    # # black_pixel_images_training, labels_training = get_origin_rectangles_training(training_images, training_labels,
    # #                                                                               label_count, n)
    # black_pixel_images_training, labels_training = get_origin_rectangles_testing(training_images, training_labels, n)
    # black_pixel_images_testing, labels_testing = get_origin_rectangles_testing(testing_images, testing_labels, n)
    #
    # # Generate 100 random rectangles
    # random_rectangles = generate_random_rectangles(n, count_random_rectangle)
    #
    # # Obtain features - horizontal and vertical features of Image
    # # Training
    # features_training = fetch_features(black_pixel_images_training, random_rectangles)
    #
    # features_training = np.array(features_training)
    # dataset = [[0] * 201 for _ in range(len(labels_training))]
    # i = -1
    # for data_point, label in zip(features_training, labels_training):
    #     # print(len(data_point))
    #     i += 1
    #     j = 0
    #     for feature_val in data_point:
    #         dataset[i][j] = float(feature_val)
    #         j += 1
    #     dataset[i][j] = int(label)
    # df = pd.DataFrame(dataset, columns=None)
    # print(df.head())
    # df.to_csv('Haar_feature_full_training.csv', header=None, index=False)
    #
    # # Obtain features - horizontal and vertical features of Image
    # # Testing
    #
    # features_testing = fetch_features(black_pixel_images_testing, random_rectangles)
    # features_testing = np.array(features_testing)
    # dataset = [[0] * 201 for _ in range(len(labels_testing))]
    # i = -1
    # for data_point, label in zip(features_testing, labels_testing):
    #     i += 1
    #     j = 0
    #     for feature_val in data_point:
    #         dataset[i][j] = float(feature_val)
    #         j += 1
    #     dataset[i][j] = int(label)
    # df = pd.DataFrame(dataset, columns=None)
    # print(df.head())
    # df.to_csv('Haar_feature_testing.csv', header=None, index=False)

    ################## Actual code after HAAR feature extraction #########################
    # Extraction was done and saved in a file
    training_data, testing_data = extract_haar_features()
    training_data = shuffle(training_data)

    # Training
    # print("Training", training_data[:, 0:5])
    # print("Training", training_data.shape)

    # Testing

    # print("Testing", testing_data[:, 0:5])
    # print("Testing", testing_data.shape)

    # ECOC

    label_code = fetch_label_code()
    print("Label shape", label_code.shape)
    original_test_y = np.copy(testing_data[:, -1])

    train_label = []
    for val in training_data[:, -1]:
        train_label.append(label_code[int(val)])
    training_data = np.delete(training_data, -1, 1)
    training_data = np.column_stack((training_data, train_label))

    print(training_data.shape)

    test_label = []
    for val in testing_data[:, -1]:
        test_label.append(label_code[int(val)])
    testing_data = np.delete(testing_data, -1, 1)
    testing_data = np.column_stack((testing_data, test_label))

    print(testing_data.shape)

    # {-1,1}
    for row in range(len(training_data)):
        for col in range(-1, -51, -1):
            if training_data[row][col] == 0.0:
                training_data[row][col] = - 1.0

    training_x = np.copy(training_data[:, 0:-50])
    testing_x = np.copy(testing_data[:, 0:-50])

    test_prediction_array = []
    train_prediction_array = []

    number_iterations = 250

    for i in range(50, 0, -1):
        training_y = np.copy(training_data[:, -i])
        testing_y = np.copy(testing_data[:, -i])
        # testing_y = []
        classifiers = adaboost_algo(training_x, training_y, testing_x, testing_y, number_iterations)
        # classifiers = adaboost_algo_random(training_x, training_y, testing_x, testing_y, number_iterations)

        #classifiers = adaboost_algo_interval(training_x, training_y, testing_x, testing_y, number_iterations)
        prediction_y_train = predict(classifiers, training_x)
        prediction_y_test = predict(classifiers, testing_x)

        # train_prediction_array.append(prediction_y_train)
        test_prediction_array.append(prediction_y_test)

        # training_accuracy = evaluate_prediction_accuracy(training_y, prediction_y_train)
        # testing_accuracy = evaluate_prediction_accuracy(testing_y, prediction_y_test)
        #
        # print("Testing accuracy for ", -i, "is:", testing_accuracy)
        # print("Testing error rate for ", -i, "is:", 1 - testing_accuracy)
        print("Round done:", i)

        # print("Training accuracy for ", -i, "is:", training_accuracy)
        # print("Training error rate for ", -i, "is:", 1 - training_accuracy)

    # Correct format - Transpose
    test_prediction = np.array(test_prediction_array).T
    train_prediction_array = np.array(train_prediction_array).T

    print(test_prediction.shape)

    for i in range(len(test_prediction)):
        for j in range(len(test_prediction[0])):
            if test_prediction[i][j] == -1:
                test_prediction[i][j] = 0

    print("Predictions are")
    print(test_prediction)
    y_label_prediction = fetch_minimum_distance(test_prediction, label_code)

    # calculating test accuracy
    print(y_label_prediction)
    print(original_test_y)
    print(len(y_label_prediction))
    print(len(original_test_y))

    testing_accuracy = evaluate_prediction_accuracy(y_label_prediction, original_test_y)

    print("Testing accuracy is", testing_accuracy)


if __name__ == '__main__':
    main()
