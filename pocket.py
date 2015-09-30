# X(5,500)
# Y(500,1)
# W(1,5)

import numpy as np
import random

TRAIN_NAME = "pocket_hw1_train.dat"
TEST_NAME = "pocket_hw1_test.dat"


def label(score):
    if score <= 0:
        return -1
    else:
        return 1


def error_rate(x, y, w):
    perceptron_score = w * x
    rate = 0
    for i in range(perceptron_score.shape[1]):
        if label(perceptron_score[0, i]) != y[i, 0]:
            rate += 1
    return float(rate) / perceptron_score.shape[1]


def pocket(x, y, update=50):

    w = np.zeros(shape=(1, 5))
    w_pocket = np.zeros(shape=(1, 5))

    update_idx = 0
    col_idx = 0
    col_number = x.shape[1]

    error_pocket = error_rate(x, y, w_pocket)

    for update_idx in range(update):
        col_range = random.sample(range(col_number), col_number)

        for col_idx in col_range:
            perceptron_score = w * x[:, col_idx]
            if label(perceptron_score) != y[col_idx, 0]:
                w = w + (x[:, col_idx] * y[col_idx, 0]).transpose()
                error = error_rate(x, y, w)
                if error < error_pocket:
                    w_pocket = w
                    error_pocket = error
                break
    return w_pocket


def main():
    train = np.loadtxt(TRAIN_NAME)
    train = np.asmatrix(train)
    test = np.loadtxt(TEST_NAME)
    test = np.asmatrix(test)

    train_x = train[:, :4].transpose()
    train_x = np.vstack((np.ones(train_x.shape[1]), train_x))
    train_y = train[:, 4:5]

    test_x = test[:, :4].transpose()
    test_x = np.vstack((np.ones(test_x.shape[1]), test_x))
    test_y = test[:, 4:5]

    iterate_number = 10
    total_rate = 0

    for i in range(iterate_number):
        w_pocket = pocket(train_x, train_y)
        total_rate += error_rate(test_x, test_y, w_pocket)

    print float(total_rate) / iterate_number

if __name__ == '__main__':
    main()
