"""
Matthew Benson
6729388
Due: Dec 5th 2021
Parity Bit Implementation (COSC 3P71)
"""

from typing import *
import math
import numpy as np
import random


def main():
    hidden_layer_amount = 8
    learning_rate = 0.6
    epoch_amount = 10000

    training_set = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    training_parity_set = [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]

    testing_set = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 0, 1],
    ]

    testing_parity_set = [1, 0, 0, 1]

    training = (training_set, training_parity_set)
    testing = (testing_set,testing_parity_set )

    w2, w3 = initialize_weights(hidden_layer_amount, 4)
    print("<---------------------->")

    # Train network
    w2, w3 = neural_network(training, epoch_amount, learning_rate, w2, w3, True)

    # Output Training examples and Output Testing Examples
    neural_network(training, 1, learning_rate, w2, w3, False)
    print("<-- New data -->")
    neural_network(testing, 1, learning_rate, w2, w3, False)


def neural_network(data_input: tuple, epochs: int, learning_rate: float, W2: np.array, W3: np.array, training_flag: bool):
    """
    Main brains of the neural network, function uses respective feed forward and back propogation techniques
    to train and adjust weights of connecting input layer, hidden layer and output layer to properly
    predict parity of given input byte.

    :param data_input: tuple of the bytes to train as well as their respective parity
    :param epochs: amount of iterations to run
    :param learning_rate: learning rate used in back propogation
    :param W2: Weights connecting A1 with A2
    :param W3: Weights connecting A2 with A3
    :param training_flag: Whether to run the network in testing mode (with new data never seen) or training mode
    :return: Weights of W2 and W3 to be used in testing mode
    """

    input_set = data_input[0]
    parity_set = data_input[1]

    for i in range(epochs):

        if training_flag:
            mean_square_error = 0

        for j in range(len(input_set)):

            """
            Feed Forward
            """

            # X1
            X1 = np.array(input_set[j])
            arr = []
            for x in range(len(X1)):
                arr.append(sig(X1[x]))

            # A1
            A1 = np.array(arr)
            A1 = np.vstack(A1)

            # X2
            X2 = np.matmul(W2, A1)
            X2 = np.vstack(X2)

            # A2
            arr = []
            for p in range(len(X2)):
                arr.append(sig(X2[p]))

            A2 = np.array(arr)
            A2 = np.vstack(A2)

            # X3
            X3 = np.matmul(W3, A2)

            # A3
            A3 = np.array(sig(X3))

            if not training_flag:
                print(input_set[j], "Expected ->", parity_set[j], "; Prediction (Rounded) ->", np.rint(A3[0]),";","Actual Output ->", A3[0])

            if training_flag:

                t = cost(A3[0], parity_set[j])
                mean_square_error += t

                """
                Back propagation
                """

                W2 = updateWeight_delta2(A3, X3, W3, W2, X2, A1, parity_set[j], learning_rate)
                W3 = updateWeight_delta3(A3, X3, A2, W3, parity_set[j], learning_rate)

        if training_flag:
            mean_square_error = mean_square_error / len(input_set)
            if (i+1) % 200 == 0:
                print("Epoch", i+1, "-> [MSE]: ", mean_square_error)

    return W2, W3


def updateWeight_delta2(A3, X3, W3, W2, X2, A1, y, learning_rate):
    """ Uses backprogration to update the weights with the new given data and expected output

    :param A3: Output layer
    :param X3: Output layer before activation
    :param W3: Weights connecting hidden and output layer
    :param W2: Weights connecting input and hidden layer
    :param X2: Hidden layer before activation
    :param A1: Input layer after activation
    :param y: Expected output, either 0 or 1
    :param learning_rate: Learning rate used in back propogation
    :return: resulting weight matrix, if 8 hidden nodes, returns 4x8 matrix
    """

    tempA = 2 * (A3[0] - y)
    tempB = np.multiply(sig(X3[0]), 1 - sig(X3[0]))
    M1 = np.array([np.multiply(tempA, tempB)])

    M2 = np.multiply(M1, W3)

    # prob WRONG LMAO
    arr = []
    for i in range(len(X2)):
        arr.append(sig(X2[i]))

    A2 = np.array(arr)
    M3 = np.multiply(A2, (1 - A2))
    M3 = np.transpose(M3)

    M3 = np.multiply(M2, M3)
    M3 = np.transpose(M3)
    #

    A1 = np.transpose(A1)

    # M4 Holds values for update
    M4 = np.matmul(M3, A1)

    delta2 = M4

    # update the weights

    M5 = np.multiply(learning_rate, delta2)
    new_weight_matrix = np.subtract(W2, M5)

    return new_weight_matrix


def updateWeight_delta3(A3, X3, A2, W3, y, learning_rate):
    """ Uses backprogration to update the weights with the new given data and expected output

    :param A3: Output node
    :param X3: Output node before activation
    :param A2: Hidden layer nodes
    :param W3: Weights connecting hidden layer and output node
    :param y: Expected Output either 0 or 1
    :param learning_rate: Learning rate used in back propogation
    :return:  resulting weight matrix, if 8 hidden nodes, returns 1x8 matrix
    """

    tempA = 2 * (A3[0] - y)
    tempB = np.multiply(sig(X3[0]), 1 - sig(X3[0]))
    M1 = np.array([np.multiply(tempA, tempB)])
    A2 = np.transpose(A2)

    # M2 Holds values for update
    M2 = np.matmul(M1, A2)

    # update the weights

    M3 = np.multiply(learning_rate, M2)
    new_weight_matrix = np.subtract(W3, M3)
    # print("breakpoint")

    return new_weight_matrix


def initialize_weights(hidden_nodes, byte_length):
    """ Initializes the two weight matrices with a starting point, complete random float values from range -1..1

    :param hidden_nodes: Amount of hidden nodes in hidden layer
    :param byte_length: Length of each byte, in our case always 4
    :return: Weight matrix for W2 and W3 respectivefully filled with totally random weights.
    """
    # initialize weights, W2
    matrix = []
    for i in range(hidden_nodes):
        row = []
        for j in range(byte_length):
            row.append(random.uniform(-1, 1))
        matrix.append(row)

    W2 = np.array(matrix)

    # initialize weights, W3
    matrix = []
    for i in range(hidden_nodes):
        matrix.append(random.uniform(-1, 1))

    W3 = np.array(matrix)

    return W2, W3


def parity(byte) -> int:
    """ Calculates the expected parity for a given byte
    :param byte: byte to be tested, example [1,1,0,1]
    :return: parity of input byte(0 or 1), example 0
    """
    one_count = 0
    for bit in byte:
        if bit == 1:
            one_count += 1
    r = one_count % 2
    return 1 if r == 0 else 0


def sig(x) -> int:
    """ Sigmoid function used to normalize into range of 0..1
    :param x: value to be used with sigmoid formula
    :return: new float value after activation
    """
    r = 1.0 / (1.0 + math.e ** (-x))
    return r


def cost(A3, y):
    """ Determines cost, used in calculating the MSE of everu epoch

    :param A3: Output layer value
    :param y: Expected output
    :return: Mean Error / Cost of expected vs outputed value
    """
    r = math.pow((A3 - y), 2)
    return r


if __name__ == '__main__':
    main()
