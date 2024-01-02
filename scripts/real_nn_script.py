import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import wrapped_numpy as wnp
from utils.synthetic_data import make_linear_multiclass_dataset


def cel(y, yh):
    return wnp.div(-wnp.axis_sum(wnp.mul(y, wnp.log(yh))), yh.shape[0])


def dcel(y, yh):
    return wnp.div(-y, wnp.mul(yh, yh.shape[0]))


def mse(y, yh):
    return wnp.div(wnp.axis_sum(wnp.pwr(wnp.sub(y, yh), 2)), 2 * y.shape[0])


def dmse(y, yh):
    return wnp.div(wnp.sub(yh, y), y.shape[0])


def relu(x):
    return wnp.where(x >= 0, x, 0.0)


def drelu(x):
    return wnp.where(x >= 0, 1.0, 0.0)


def softmax(x):
    maximums = np.max(x, axis=1).reshape(-1, 1)
    exponents = wnp.exp(wnp.sub(x, maximums))
    sums = wnp.axis_sum(exponents, axis=1).reshape(-1, 1)

    return wnp.div(exponents, sums)


def dsoftmax(x):
    return wnp.mul(softmax(x), wnp.sub(1, softmax(x)))


def sigmoid(x):
    return wnp.where(
        x >= 0,
        wnp.div(1, wnp.add(1, wnp.exp(-x))),
        wnp.div(wnp.exp(x), wnp.add(1, wnp.exp(x))),
    )


def dsigmoid(x):
    return wnp.mul(sigmoid(x), wnp.sub(1, sigmoid(x)))


def ohe(y):
    y = y.ravel().astype(np.int_)
    num_classes = wnp.unique(y).shape[0]

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
        t1 = wnp.add(wnp.dot(x, w1), b1)
        # a1 = relu(t1)
        a1 = wnp.tanh(t1)

        t2 = wnp.add(wnp.dot(a1, w2), b2)
        # yh = softmax(t2)
        yh = sigmoid(t2)

        # loss.append(cel(y, yh))
        loss.append(mse(y, yh))
        # dyh = dcel(y, yh)
        dyh = dmse(y, yh)

        # dt2 = dsoftmax(t2) * dyh
        dt2 = wnp.mul(dsigmoid(t2), dyh)
        dw2 = wnp.dot(wnp.transpose(a1), dt2)
        db2 = wnp.axis_sum(dt2, axis=0)
        da1 = wnp.dot(dt2, wnp.transpose(w2))

        # dt1 = drelu(t1) * da1
        dt1 = wnp.mul(wnp.sub(1, wnp.pwr(wnp.tanh(t1), 2)), da1)
        dw1 = wnp.dot(wnp.transpose(x), dt1)
        db1 = wnp.axis_sum(dt1, axis=0)

        w2 = wnp.sub(w2, wnp.mul(n, dw2))
        b2 = wnp.sub(b2, wnp.mul(n, db2))

        w1 = wnp.sub(w1, wnp.mul(n, dw1))
        b1 = wnp.sub(b1, wnp.mul(n, db1))

    plt.plot(loss)
    plt.show()

    print(y[:5])
    print(yh[:5])


if __name__ == "__main__":
    main()
