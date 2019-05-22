import numpy as np
import sys

MALE = 0.02
FEMALE = 0.06
INFANT = 0.1
ETA = 0.01
EPOCH = 5


# shuffle the data x and y together.
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


# find the index that is not y or y_hat from 0-2
def get_other_y(y, y_hat):
    if (y == 0 and y_hat == 1) or (y == 1 and y_hat == 0):
        return 2
    if (y == 0 and y_hat == 2) or (y == 2 and y_hat == 0):
        return 1
    if (y == 2 and y_hat == 1) or (y == 1 and y_hat == 2):
        return 0


# pa update algorithm
def pa(w, x, y, i, hat):
    if np.count_nonzero(x[i]) == 0:
        return

    tao = (max(0, 1 - np.dot(w[y[i]], x[i]) + np.dot(w[y[hat]], x[i]))) / (2 * (np.linalg.norm(x[i]) ** 2))

    w[y[i]] = w[y[i]] + x[i] * tao
    w[hat] = w[hat] - x[i] * tao

    return w


# svm update algorithm
def svm(w, x, y, i, hat):
    lamda = 0.01
    eta = ETA
    other = get_other_y(i, hat)
    w[y[i]] = (1 - lamda * eta) * w[y[i]] + x[i] * eta
    w[hat] = (1 - lamda * eta) * w[hat] - x[i] * eta
    w[other] = (1 - lamda * eta) * w[other]

    return w


# perceptron update algorithm
def perceptron(w, x, y, i, hat):
    eta = ETA
    w[y[i]] = w[y[i]] + x[i] * eta
    w[hat] = w[hat] - x[i] * eta

    return w


# read the train_y file
def read_train_y(file_name, y):
    with open(file_name) as fp:
        line = fp.readline()
        while line:
            y.append(int(line.strip()[0]))
            line = fp.readline()

    return np.array(y)


# read train_x file.
def read_train_x(file_name, x):
    with open(file_name) as fp:
        line = fp.readline()
        count = 1
        while line:
            line = line.strip()
            x.append(line.split(","))
            if x[count][0] == "M":
                x[count][0] = MALE
            elif x[count][0] == "F":
                x[count][0] = FEMALE
            else:
                x[count][0] = INFANT
            count = count + 1
            line = fp.readline()
    x.pop(0)

    x = np.array(x).astype(np.float)

    return x


# norm the data by min max.
def min_max_norm(x_matrix):
    col_num = len(x_matrix[0])
    for i in range(0, col_num):
        x_matrix[:, [i]] = (x_matrix[:, [i]] - x_matrix[:, [i]].min()) / \
                           (x_matrix[:, [i]].max() - x_matrix[:, [i]].min())


# test w on the 10% data.
def test_loss(x_test, y_test, w):
    temp_w = w
    size = len(x_test)
    count_loss = 0
    # test the loss on the new 10% data.
    for t in range(0, size):
        y_hat = np.argmax(np.dot(temp_w, x_test[t]))
        if y_test[t] != y_hat:
            count_loss = count_loss + 1

    if size == 0:
        return 0.0
    return float(count_loss) / size


# the function train w on the training data.
# the function splits to two parts, 90% the training data and 10% minor weight test. 90% will train 20 weights,
# and will check the loss of every weight on 10% of the minor test data, and return the weight with the lowest loss.
# EPOCH - the number of time the weight update on the 90% data.
def train_algorithms(x, y, algo):
    dc = len(x[0])
    num_of_classification = 3
    num_of_weights = 20

    w = np.zeros((num_of_weights, num_of_classification, dc))
    loss = np.zeros(num_of_weights)
    # create 'num_of_weights' weights and choose the best one.
    for i in range(0, num_of_weights):
        shuffle_in_unison(x, y)
        # train on 90% of the data.
        x, x_test = np.split(x, [int(len(x) * (9 / 10))])
        y, y_test = np.split(y, [int(len(y) * (9 / 10))])
        # run 'EPOCH' times on the 90% train data.
        for epoch in range(0, EPOCH):
            # run over the 90% data.
            for t in range(0, len(x)):
                y_hat = np.argmax(np.dot(w[i], x[t]))
                if y_hat != y[t]:
                    w[i] = algo(w[i], x, y, t, y_hat)

        # test w on the 10% data.
        loss[i] = test_loss(x_test, y_test, w[i])
    # return the weight with the lowest loss.
    return w[np.argmin(loss)]


# test one weight.
def test_w(x, w):
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = np.argmax(np.dot(w, x[i]))

    return result


# test all the weights of perseptron , svm and pa
# w[0] = perseptron , w[1] = svm , w[2] = pa
def test_all(x, w):
    result = np.zeros((3, len(x)))

    result[0] = test_w(x, w[0])
    result[1] = test_w(x, w[1])
    result[2] = test_w(x, w[2])

    return result


def print_all(result):

    for i in range(len(result[0])):
        print("perceptron: ", end='')
        print(int(result[0][i]), end='')
        print(", svm: ", end='')
        print(int(result[1][i]), end='')
        print(", pa: ", end='')
        print(int(result[2][i]))


def main():
    y = []
    train_x = [[]]
    test_x = [[]]
    # read fro, files.
    train_x = read_train_x(sys.argv[1], train_x)
    test_x = read_train_x(sys.argv[3], test_x)
    y = read_train_y(sys.argv[2], y)
    # norm the data, the norm give bed loss.
    # min_max_norm(train_x)
    # min_max_norm(test_x)
    # train.
    w_perceptron = train_algorithms(train_x, y, perceptron)
    w_svm = train_algorithms(train_x, y, svm)
    w_pa = train_algorithms(train_x, y, pa)
    # test.
    w_all = [w_perceptron, w_svm, w_pa]
    result = test_all(test_x, w_all)
    # print.
    print_all(result)


if __name__ == "__main__":
    main()
