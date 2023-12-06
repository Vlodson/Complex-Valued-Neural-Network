import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.synthetic_data import (
    make_linear_binary_dataset,
    make_linear_multiclass_dataset,
)


def cel(y, yh):
    return -np.sum(y * np.log(yh)) / yh.shape[0]


def dcel(y, yh):
    return -y / (yh * yh.shape[0])


def mse(y, yh):
    return np.sum((y - yh) ** 2) / (2 * y.shape[0])


def dmse(y, yh):
    return (yh - y) / y.shape[0]


def relu(x):
    return np.where(x >= 0, x, 0.0)


def drelu(x):
    return np.where(x >= 0, 1.0, 0.0)


def softmax(x):
    maximums = np.max(x, axis=1).reshape(-1, 1)
    exponents = np.exp(x - maximums)
    sums = np.sum(exponents, axis=1).reshape(-1, 1)

    return exponents / sums


def dsoftmax(x):
    return softmax(x) * (1 - softmax(x))


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ohe(y):
    y = y.ravel().astype(np.int_)
    num_classes = np.unique(y).shape[0]

    one_hot_encoded = np.zeros((y.shape[0], num_classes))
    one_hot_encoded[np.arange(y.shape[0]), y] = 1

    return one_hot_encoded


def main():
    x, y = make_linear_multiclass_dataset()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

    y = ohe(y.ravel())

    w1 = np.random.uniform(-1, 1, (2, 3))
    b1 = np.random.uniform(-1, 1, (1, 3))

    w2 = np.random.uniform(-1, 1, (3, 3))
    b2 = np.random.uniform(-1, 1, (1, 3))

    loss = []
    n = 1e-1

    for _ in tqdm(range(10000)):
        t1 = x @ w1 + b1
        # a1 = relu(t1)
        a1 = np.tanh(t1)

        t2 = a1 @ w2 + b2
        # yh = softmax(t2)
        yh = sigmoid(t2)

        # loss.append(cel(y, yh))
        loss.append(mse(y, yh))
        # dyh = dcel(y, yh)
        dyh = dmse(y, yh)

        # dt2 = dsoftmax(t2) * dyh
        dt2 = dsigmoid(t2) * dyh
        dw2 = a1.T @ dt2
        db2 = np.sum(dt2, axis=0)
        da1 = dt2 @ w2.T

        # dt1 = drelu(t1) * da1
        dt1 = (1 - np.tanh(t1) ** 2) * da1
        dw1 = x.T @ dt1
        db1 = np.sum(dt1, axis=0)

        w2 -= n * dw2
        b2 -= n * db2

        w1 -= n * dw1
        b1 -= n * db1

    plt.plot(loss)
    plt.show()

    print(y[:5])
    print(yh[:5])


if __name__ == "__main__":
    main()
