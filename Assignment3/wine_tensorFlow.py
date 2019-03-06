# Citation - https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from random import shuffle, seed


def extract_dataset():
    training_url = "train_wine.csv"
    testing_url = "test_wine.csv"
    training_data = pd.read_csv(training_url, header=None, sep=',' )
    testing_data = pd.read_csv(testing_url, header=None, sep=',')
    return training_data, testing_data


def dense_to_one_hot(y, num_of_units):
    a = tf.one_hot(y, num_of_units).eval()
    return a


def main():
    seed = 1
    rng = np.random.RandomState(seed)

    training, testing = extract_dataset()
    shuffle(training.values)
    shuffle(testing.values)
    LR = 0.01
    epochs = 750

    X_train = training.iloc[:, 1:training.shape[1]].values
    X_test = testing.iloc[:, 1:testing.shape[1]].values
    Y_train = training.iloc[:, 0].values
    Y_test = testing.iloc[:, 0].values

    # Y_train = Y_train.reshape(-1, 1)
    # Y_test = Y_test.reshape(-1, 1)

    #print(X_train.shape, Y_train.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    input_num_units = 13
    hidden_number = 3
    output_num_units = 3

    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])

    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_number], seed=seed)),
        'output': tf.Variable(tf.random_normal([hidden_number, output_num_units], seed=seed))
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_number], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }

    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.sigmoid(hidden_layer)

    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

    loss = tf.reduce_mean(tf.squared_difference(output_layer, y))

    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # create initialized variables
        sess.run(init)

        for epoch in range(epochs):
            _, c, a = sess.run([optimizer, loss, accuracy], feed_dict={x: X_train, y: dense_to_one_hot(Y_train, 3)})
            print("Epoch:", (epoch + 1), "Loss =", c, "Accuracy =", a)
            if a == 1.0:
                break

        print("Training complete")
        #print("Training accuracy is", a*100)

        print("Testing accuracy is ", accuracy.eval({x: X_test.reshape(-1, input_num_units), y: dense_to_one_hot(Y_test, 3)})*100)

        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({x: X_test.reshape(-1, input_num_units)})


if __name__ == '__main__':
    main()