import numpy as np


def label(score):
    if score <= 0:
        return -1
    else:
        return 1


def perceptron1(X, Y, W, count):

    count[0] = 0
    has_error = False
    col_idx = 0
    col_number = X.shape[1]
    update = 0

    while True:
        print count[0]
        if col_idx == col_number - 1:
            if not has_error:
                break
            col_idx = 0
            has_error = False

        perceptron_score = np.dot(W, X[:, col_idx])
        sign_result = label(perceptron_score)

        if sign_result != Y[col_idx]:
            has_error = True
            update += 1
            W += np.dot(X[:, col_idx], Y[col_idx]).transpose()

        count[0] += 1
        col_idx += 1
    return update


def main():
    data = np.loadtxt("machine_learning_hw1.dat")
    data = np.asmatrix(data)

    X = data[:, :4].transpose()
    X = np.vstack((X, np.ones(X.shape[1])))
    Y = data[:, 4:5]
    W = np.zeros(shape=(1, 5))

    count = [0]
    print perceptron1(X, Y, W, count)
    print "Update Count is:", count[0]

if __name__ == '__main__':
    main()
