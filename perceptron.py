import numpy as np
import random


def label(score):
    if score <= 0:
        return -1
    else:
        return 1


def perceptron(x, y, w, count, rand=False, alpha=1):

    count[0] = 0
    has_error = False
    idx = 0
    col_idx = 0
    col_number = x.shape[1]
    col_range = range(col_number) if not rand \
        else random.sample(range(col_number), col_number)
    update = 0

    while True:
        if idx == col_number:
            if not has_error:
                break
            idx = 0
            has_error = False

        col_idx = col_range[idx]

        perceptron_score = np.dot(w, x[:, col_idx])
        sign_result = label(perceptron_score)

        if sign_result != y[col_idx]:
            has_error = True
            update += 1
            w += alpha * np.dot(x[:, col_idx], y[col_idx]).transpose()

        count[0] += 1
        idx += 1
    return update


def main():
    data = np.loadtxt("machine_learning_hw1.dat")
    data = np.asmatrix(data)

    x = data[:, :4].transpose()
    x = np.vstack((x, np.ones(x.shape[1])))
    y = data[:, 4:5]
    w = np.zeros(shape=(1, 5))

    # Q15
    # count = [0]
    # print "1:Update Count is:", perceptron(x, y, w, count)
    # print "1:Iterative Count is:", count[0]

    # Q16
    # count = [0]
    # total_count = 0
    # for i in xrange(0, 2000):
    #     w = np.zeros(shape=(1, 5))
    #     total_count = total_count + perceptron(x, y, w, count, True)
    # total_count /= 2000
    # print "2:Update Count is:", total_count
    # print "2:Iterative Count is:", count[0]

    # Q17
    # count = [0]
    # total_count = 0
    # for i in xrange(0, 2000):
    #     w = np.zeros(shape=(1, 5))
    #     total_count = total_count + perceptron(x, y, w, count, True, 0.5)
    # total_count /= 2000
    # print "3:Update Count is:", total_count
    # print "3:Iterative Count is:", count[0]


if __name__ == '__main__':
    main()
