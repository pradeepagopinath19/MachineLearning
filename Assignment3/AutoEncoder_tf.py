# Citation - https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from random import shuffle, seed


def extract_dataset():
    url = "Auto_encoder.csv"
    data = pd.read_csv(url, header=None, sep=',')
    return data


def dense_to_one_hot(y, num_of_units):
    print(y.shape)
    a = tf.one_hot(y, num_of_units).eval()
    print(a.shape)
    return a


def main():
    seed = 1
    rng = np.random.RandomState(seed)

    data = extract_dataset()

    LR = 0.01
    epochs = 4500

    X_train = X_test = Y_train = Y_test = data.values


    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    input_num_units = 8
    hidden_number = 3
    output_num_units = 8

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
            _, c, a = sess.run([optimizer, loss, accuracy], feed_dict={x: X_train, y: Y_train})
            print("Epoch:", (epoch + 1), "Loss =", c)
            if a == 1.0:
                print(epoch)
                break
        print("The accuracy is", a)

        print("Training complete")

        # pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        #print("Test Accuracy:", accuracy.eval({x: X_test.reshape(-1, input_num_units), y: Y_test})*100)


        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({x: X_test.reshape(-1, input_num_units)})


if __name__ == '__main__':
    main()
